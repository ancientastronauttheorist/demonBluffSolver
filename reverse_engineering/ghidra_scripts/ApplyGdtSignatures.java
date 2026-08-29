// Applies exact C function definitions from a read-only GDT archive.
// @category DemonBluff

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

import ghidra.app.cmd.function.ApplyFunctionSignatureCmd;
import ghidra.app.cmd.function.FunctionRenameOption;
import ghidra.app.util.headless.HeadlessScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.data.DataType;
import ghidra.program.model.data.DataTypeConflictHandler;
import ghidra.program.model.data.FileDataTypeManager;
import ghidra.program.model.data.FunctionDefinition;
import ghidra.program.model.data.FunctionDefinitionDataType;
import ghidra.program.model.data.ParameterDefinition;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Parameter;
import ghidra.program.model.listing.VariableStorage;
import ghidra.program.model.symbol.SourceType;
import ghidra.program.model.symbol.Symbol;
import ghidra.program.model.symbol.SymbolUtilities;

public class ApplyGdtSignatures extends HeadlessScript {
    private static final int SUMMARY_SCHEMA_VERSION = 1;
    private static final String WINDOWS_X64_CALLING_CONVENTION = "__fastcall";
    private static final Pattern DECLARED_IDENTIFIER = Pattern.compile(
        "([A-Za-z_][A-Za-z0-9_]*)\\s*$"
    );
    private static final String[] WINDOWS_X64_ARGUMENT_REGISTERS = {
        "RCX", "RDX", "R8", "R9"
    };
    private static final int[] SEVEN_ARGUMENT_STACK_OFFSETS = { 0x28, 0x30, 0x38 };

    @Override
    protected void run() throws Exception {
        Path summaryPath = null;
        Path summaryTempPath = null;
        try {
            if (!isRunningHeadless()) {
                throw new IllegalStateException(
                    "ApplyGdtSignatures.java must run as a headless pre-script"
                );
            }

            String[] args = getScriptArgs();
            if (args.length != 3) {
                throw new IllegalArgumentException(
                    "Expected three arguments: GDT path, target JSON path, completion-summary path"
                );
            }

            Path gdtPath = Path.of(args[0]).toAbsolutePath().normalize();
            Path targetPath = Path.of(args[1]).toAbsolutePath().normalize();
            summaryPath = Path.of(args[2]).toAbsolutePath().normalize();
            summaryTempPath = temporarySummaryPath(summaryPath);
            requireDistinctOutput(summaryPath, summaryTempPath, gdtPath, targetPath);
            Files.deleteIfExists(summaryPath);
            Files.deleteIfExists(summaryTempPath);
            requireRegularFile(gdtPath, "GDT archive");
            requireRegularFile(targetPath, "target JSON");
            validateWindowsX64Program();

            TargetSet targetSet = readTargets(targetPath);
            if (targetSet.targets.isEmpty()) {
                throw new IllegalArgumentException(
                    "Target set contains no functions: " + targetPath
                );
            }

            int applied = 0;
            int validated = 0;
            int sevenArgumentTargets = 0;
            int preservedLabels = 0;
            int dataTypeCountBefore = currentProgram.getDataTypeManager()
                .getDataTypeCount(true);
            try (FileDataTypeManager archive = FileDataTypeManager.openFileArchive(
                    gdtPath.toFile(), false)) {
                if (archive.isUpdatable()) {
                    throw new IllegalStateException(
                        "GDT archive was not opened read-only: " + gdtPath
                    );
                }
                if (archive.getDataOrganization().getPointerSize() != 8) {
                    throw new IllegalArgumentException(
                        "GDT archive is not configured for 64-bit pointers: " + gdtPath
                    );
                }

                List<PreparedTarget> preparedTargets = prepareTargets(
                    targetSet.targets, archive
                );
                monitor.checkCancelled();

                for (PreparedTarget prepared : preparedTargets) {
                    monitor.checkCancelled();
                    applySignature(prepared);
                    applied++;
                }

                for (PreparedTarget prepared : preparedTargets) {
                    monitor.checkCancelled();
                    validateAppliedTarget(prepared);
                    if (prepared.archiveDefinition.getArguments().length == 7) {
                        validateSevenArgumentWindowsX64Storage(prepared);
                        sevenArgumentTargets++;
                    }
                    preservedLabels += prepared.labelsBefore.size();
                    validated++;
                }
            }

            int dataTypeCountAfter = currentProgram.getDataTypeManager()
                .getDataTypeCount(true);
            if (dataTypeCountAfter < dataTypeCountBefore) {
                throw new IllegalStateException(
                    "Applying GDT signatures unexpectedly removed program datatypes"
                );
            }
            int importedDataTypes = dataTypeCountAfter - dataTypeCountBefore;
            monitor.checkCancelled();
            end(true);
            start();
            println(
                "GDT signatures applied and validated: applied=" + applied +
                ", validated=" + validated + ", imported_datatypes=" + importedDataTypes
            );
            writeSummaryAtomically(
                summaryPath,
                summaryTempPath,
                targetSet,
                applied,
                validated,
                importedDataTypes,
                sevenArgumentTargets,
                preservedLabels
            );
        }
        catch (Throwable failure) {
            removeIncompleteSummary(summaryTempPath, failure);
            removeIncompleteSummary(summaryPath, failure);
            rollbackScriptTransactionAndAbort(failure);
            rethrowFailure(failure);
        }
    }

    private Path temporarySummaryPath(Path summaryPath) {
        return summaryPath.resolveSibling(summaryPath.getFileName() + ".tmp");
    }

    private void requireDistinctOutput(
            Path summaryPath,
            Path summaryTempPath,
            Path gdtPath,
            Path targetPath) {
        if (summaryPath.equals(gdtPath) || summaryPath.equals(targetPath) ||
                summaryTempPath.equals(gdtPath) || summaryTempPath.equals(targetPath)) {
            throw new IllegalArgumentException(
                "Completion-summary paths must not overlap the GDT or target JSON inputs"
            );
        }
    }

    private void removeIncompleteSummary(Path path, Throwable failure) {
        if (path == null) {
            return;
        }
        try {
            Files.deleteIfExists(path);
        }
        catch (Throwable cleanupFailure) {
            failure.addSuppressed(cleanupFailure);
        }
    }

    private void rollbackScriptTransactionAndAbort(Throwable failure) {
        try {
            end(false);
        }
        catch (Throwable rollbackFailure) {
            failure.addSuppressed(rollbackFailure);
        }
        try {
            start();
        }
        catch (Throwable restartFailure) {
            failure.addSuppressed(restartFailure);
        }
        try {
            setHeadlessContinuationOption(HeadlessContinuationOption.ABORT);
        }
        catch (Throwable abortFailure) {
            failure.addSuppressed(abortFailure);
        }
    }

    private void rethrowFailure(Throwable failure) throws Exception {
        if (failure instanceof Exception) {
            throw (Exception) failure;
        }
        if (failure instanceof Error) {
            throw (Error) failure;
        }
        throw new RuntimeException(failure);
    }

    private void requireRegularFile(Path path, String description) {
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException(description + " not found: " + path);
        }
    }

    private void validateWindowsX64Program() {
        String languageId = currentProgram.getLanguageID().getIdAsString();
        String compilerSpecId = currentProgram.getCompilerSpec()
            .getCompilerSpecID()
            .getIdAsString();
        if (!languageId.startsWith("x86:LE:64:") || currentProgram.getDefaultPointerSize() != 8) {
            throw new IllegalStateException(
                "Program is not x86-64 little-endian: language=" + languageId
            );
        }
        if (!compilerSpecId.toLowerCase().contains("windows")) {
            throw new IllegalStateException(
                "Program does not use a Windows compiler specification: " + compilerSpecId
            );
        }
        if (currentProgram.getCompilerSpec().getCallingConvention(
            WINDOWS_X64_CALLING_CONVENTION
        ) == null) {
            throw new IllegalStateException(
                "Windows x64 calling convention is unavailable: " +
                WINDOWS_X64_CALLING_CONVENTION
            );
        }
    }

    private List<PreparedTarget> prepareTargets(
            List<Target> targets, FileDataTypeManager archive) throws Exception {
        List<PreparedTarget> prepared = new ArrayList<>();
        Set<Long> seenRvas = new HashSet<>();
        Set<String> seenTargetNames = new HashSet<>();
        Set<String> seenDefinitionPaths = new HashSet<>();
        for (Target target : targets) {
            monitor.checkCancelled();
            if (!seenRvas.add(target.rva)) {
                throw new IllegalArgumentException("Duplicate target RVA: " + target.rvaText);
            }
            if (!seenTargetNames.add(target.name)) {
                throw new IllegalArgumentException("Duplicate target name: " + target.name);
            }

            Address address;
            try {
                address = currentProgram.getImageBase().add(target.rva);
            }
            catch (Exception exception) {
                throw new IllegalArgumentException(
                    "Invalid target RVA " + target.rvaText + ": " + exception.getMessage(),
                    exception
                );
            }
            if (!currentProgram.getMemory().contains(address)) {
                throw new IllegalArgumentException(
                    "Target RVA is outside program memory: " + target.rvaText
                );
            }
            Function function = currentProgram.getFunctionManager().getFunctionAt(address);
            if (function == null || !function.getEntryPoint().equals(address)) {
                throw new IllegalStateException(
                    "No exact function entrypoint at RVA " + target.rvaText
                );
            }

            Set<String> labelsBefore = labelsAt(address);
            String expectedMetadataLabel = safeName(target.metadataName);
            if (!labelsBefore.contains(expectedMetadataLabel)) {
                throw new IllegalStateException(
                    "Expected imported metadata label " + expectedMetadataLabel +
                    " is missing at RVA " + target.rvaText
                );
            }

            String definitionName = declaredIdentifier(target.signature);
            FunctionDefinition definition = uniqueDefinition(archive, definitionName);
            // ApplyFunctionSignatureCmd rewrites ParameterDefinition datatypes while resolving
            // dependencies, so never hand it a DB-backed definition from the read-only GDT.
            FunctionDefinition detachedDefinition = new FunctionDefinitionDataType(definition);
            String definitionPath = definition.getDataTypePath().toString();
            if (!seenDefinitionPaths.add(definitionPath)) {
                throw new IllegalArgumentException(
                    "FunctionDefinition is reused by multiple targets: " + definitionPath
                );
            }
            prepared.add(
                new PreparedTarget(
                    target,
                    address,
                    function,
                    function.getName(true),
                    labelsBefore,
                    definition,
                    detachedDefinition
                )
            );
        }
        return prepared;
    }

    private FunctionDefinition uniqueDefinition(
            FileDataTypeManager archive, String definitionName) {
        List<DataType> matches = new ArrayList<>();
        archive.findDataTypes(definitionName, matches);
        List<FunctionDefinition> definitions = new ArrayList<>();
        for (DataType match : matches) {
            if (match instanceof FunctionDefinition && match.getName().equals(definitionName)) {
                definitions.add((FunctionDefinition) match);
            }
        }
        if (definitions.size() != 1) {
            List<String> paths = new ArrayList<>();
            for (FunctionDefinition definition : definitions) {
                paths.add(definition.getDataTypePath().toString());
            }
            throw new IllegalArgumentException(
                "Expected exactly one FunctionDefinition named " + definitionName +
                ", found " + definitions.size() + ": " + paths
            );
        }
        return definitions.get(0);
    }

    private String declaredIdentifier(String signature) {
        int openParenthesis = signature.indexOf('(');
        if (openParenthesis <= 0) {
            throw new IllegalArgumentException("Invalid C signature: " + signature);
        }
        Matcher matcher = DECLARED_IDENTIFIER.matcher(
            signature.substring(0, openParenthesis)
        );
        if (!matcher.find()) {
            throw new IllegalArgumentException(
                "Could not extract declared C identifier from signature: " + signature
            );
        }
        return matcher.group(1);
    }

    private void applySignature(PreparedTarget prepared) throws Exception {
        Function function = currentProgram.getFunctionManager().getFunctionAt(prepared.address);
        if (function == null || !function.getEntryPoint().equals(prepared.address)) {
            throw new IllegalStateException(
                prepared.target.name + ": exact function entrypoint disappeared before apply"
            );
        }

        function.setCallingConvention(WINDOWS_X64_CALLING_CONVENTION);
        ApplyFunctionSignatureCmd command = new ApplyFunctionSignatureCmd(
            prepared.address,
            prepared.applyDefinition,
            SourceType.IMPORTED,
            true,
            false,
            DataTypeConflictHandler.REPLACE_EMPTY_STRUCTS_OR_RENAME_AND_ADD_HANDLER,
            FunctionRenameOption.NO_CHANGE
        );
        if (!command.applyTo(currentProgram, monitor)) {
            throw new IllegalStateException(
                prepared.target.name + ": ApplyFunctionSignatureCmd failed: " +
                command.getStatusMsg()
            );
        }

        function = currentProgram.getFunctionManager().getFunctionAt(prepared.address);
        if (function == null || !function.getEntryPoint().equals(prepared.address)) {
            throw new IllegalStateException(
                prepared.target.name + ": exact function entrypoint disappeared after apply"
            );
        }
        ParameterDefinition[] definitions = prepared.applyDefinition.getArguments();
        Parameter[] parameters = function.getParameters();
        if (parameters.length != definitions.length) {
            throw new IllegalStateException(
                prepared.target.name + ": command applied " + parameters.length +
                " parameters, expected " + definitions.length
            );
        }
        for (int index = 0; index < definitions.length; index++) {
            String name = definitions[index].getName();
            if (name != null && !name.isBlank()) {
                if (parameters[index].getSource() != SourceType.IMPORTED) {
                    // SymbolDB ignores a same-name source-only update. A temporary imported
                    // rename lets the final name retain IMPORTED provenance.
                    String temporaryName = name + "__apply_gdt_" +
                        Long.toUnsignedString(prepared.target.rva, 16) + "_" + index;
                    parameters[index].setName(temporaryName, SourceType.IMPORTED);
                }
                parameters[index].setName(name, SourceType.IMPORTED);
            }
        }
        function.setNoReturn(prepared.applyDefinition.hasNoReturn());
    }

    private void validateAppliedTarget(PreparedTarget prepared) {
        Function function = currentProgram.getFunctionManager().getFunctionAt(prepared.address);
        if (function == null || !function.getEntryPoint().equals(prepared.address)) {
            throw new IllegalStateException(
                prepared.target.name + ": exact function entrypoint disappeared"
            );
        }
        if (!function.getName(true).equals(prepared.functionNameBefore)) {
            throw new IllegalStateException(
                prepared.target.name + ": function label changed from " +
                prepared.functionNameBefore + " to " + function.getName(true)
            );
        }
        Set<String> labelsAfter = labelsAt(prepared.address);
        if (!labelsAfter.containsAll(prepared.labelsBefore)) {
            Set<String> missing = new LinkedHashSet<>(prepared.labelsBefore);
            missing.removeAll(labelsAfter);
            throw new IllegalStateException(
                prepared.target.name + ": existing labels were removed: " + missing
            );
        }
        if (!WINDOWS_X64_CALLING_CONVENTION.equals(function.getCallingConventionName())) {
            throw new IllegalStateException(
                prepared.target.name + ": expected calling convention " +
                WINDOWS_X64_CALLING_CONVENTION + ", got " +
                function.getCallingConventionName()
            );
        }
        if (function.hasCustomVariableStorage()) {
            throw new IllegalStateException(
                prepared.target.name + ": function retained custom parameter storage"
            );
        }
        if (function.getSignatureSource() != SourceType.IMPORTED) {
            throw new IllegalStateException(
                prepared.target.name + ": signature source is not IMPORTED"
            );
        }

        FunctionDefinition expected = prepared.applyDefinition;
        if (!function.getReturnType().isEquivalent(expected.getReturnType())) {
            throw new IllegalStateException(
                prepared.target.name + ": return type does not match GDT definition"
            );
        }
        if (function.hasVarArgs() != expected.hasVarArgs()) {
            throw new IllegalStateException(
                prepared.target.name + ": varargs setting does not match GDT definition"
            );
        }
        if (function.hasNoReturn() != expected.hasNoReturn()) {
            throw new IllegalStateException(
                prepared.target.name + ": no-return setting does not match GDT definition"
            );
        }

        ParameterDefinition[] expectedArguments = expected.getArguments();
        Parameter[] actualParameters = function.getParameters();
        if (function.getAutoParameterCount() != 0) {
            throw new IllegalStateException(
                prepared.target.name + ": unexpected auto-parameters were injected"
            );
        }
        if (actualParameters.length != expectedArguments.length) {
            throw new IllegalStateException(
                prepared.target.name + ": expected " + expectedArguments.length +
                " parameters, got " + actualParameters.length
            );
        }
        for (int index = 0; index < expectedArguments.length; index++) {
            ParameterDefinition expectedArgument = expectedArguments[index];
            Parameter actualParameter = actualParameters[index];
            if (!actualParameter.getFormalDataType().isEquivalent(
                expectedArgument.getDataType()
            )) {
                throw new IllegalStateException(
                    prepared.target.name + ": parameter " + index +
                    " datatype does not match GDT definition"
                );
            }
            String expectedName = expectedArgument.getName();
            if (expectedName != null && !expectedName.isBlank() &&
                    !Objects.equals(expectedName, actualParameter.getName())) {
                throw new IllegalStateException(
                    prepared.target.name + ": parameter " + index + " expected name " +
                    expectedName + ", got " + actualParameter.getName()
                );
            }
            if (expectedName != null && !expectedName.isBlank() &&
                    actualParameter.getSource() != SourceType.IMPORTED) {
                throw new IllegalStateException(
                    prepared.target.name + ": parameter " + index +
                    " name source is not IMPORTED"
                );
            }
        }
    }

    private void validateSevenArgumentWindowsX64Storage(PreparedTarget prepared) {
        Function function = currentProgram.getFunctionManager().getFunctionAt(prepared.address);
        if (function == null || !function.getEntryPoint().equals(prepared.address)) {
            throw new IllegalStateException(
                prepared.target.name + ": exact function entrypoint disappeared"
            );
        }
        Parameter[] parameters = function.getParameters();
        if (parameters.length != 7) {
            throw new IllegalStateException(
                prepared.target.name + ": seven-argument validation received " +
                parameters.length + " parameters"
            );
        }
        for (int index = 0; index < WINDOWS_X64_ARGUMENT_REGISTERS.length; index++) {
            VariableStorage storage = parameters[index].getVariableStorage();
            if (!storage.isRegisterStorage() || storage.getRegister() == null ||
                    !WINDOWS_X64_ARGUMENT_REGISTERS[index].equals(
                        storage.getRegister().getBaseRegister().getName()
                    )) {
                throw new IllegalStateException(
                    prepared.target.name + ": parameter " + index +
                    " expected register family " +
                    WINDOWS_X64_ARGUMENT_REGISTERS[index] + ", got " + storage
                );
            }
        }
        for (int stackIndex = 0; stackIndex < SEVEN_ARGUMENT_STACK_OFFSETS.length; stackIndex++) {
            int parameterIndex = WINDOWS_X64_ARGUMENT_REGISTERS.length + stackIndex;
            VariableStorage storage = parameters[parameterIndex].getVariableStorage();
            int expectedOffset = SEVEN_ARGUMENT_STACK_OFFSETS[stackIndex];
            if (!storage.isStackStorage() || storage.getStackOffset() != expectedOffset) {
                throw new IllegalStateException(
                    prepared.target.name + ": parameter " + parameterIndex +
                    " expected stack offset 0x" + Integer.toHexString(expectedOffset) +
                    ", got " + storage
                );
            }
        }
    }

    private Set<String> labelsAt(Address address) {
        Set<String> labels = new LinkedHashSet<>();
        for (Symbol symbol : currentProgram.getSymbolTable().getSymbols(address)) {
            labels.add(symbol.getName(true));
        }
        return labels;
    }

    private String safeName(String value) {
        String result = SymbolUtilities.replaceInvalidChars(value, true);
        if (result.length() > 512) {
            result = result.substring(0, 512);
        }
        return result;
    }

    private TargetSet readTargets(Path targetPath) throws Exception {
        int schemaVersion = -1;
        String buildId = null;
        List<Target> targets = new ArrayList<>();
        try (BufferedReader buffered = Files.newBufferedReader(targetPath, StandardCharsets.UTF_8);
                JsonReader reader = new JsonReader(buffered)) {
            reader.beginObject();
            while (reader.hasNext()) {
                String section = reader.nextName();
                switch (section) {
                    case "schema_version":
                        schemaVersion = reader.nextInt();
                        break;
                    case "build_id":
                        buildId = reader.nextString();
                        break;
                    case "functions":
                        readTargetFunctions(reader, targetPath, targets);
                        break;
                    default:
                        reader.skipValue();
                        break;
                }
            }
            reader.endObject();
        }
        if (schemaVersion != 1) {
            throw new IllegalArgumentException(
                "Unsupported target schema_version " + schemaVersion + " in " + targetPath
            );
        }
        if (buildId == null || buildId.isBlank()) {
            throw new IllegalArgumentException("Target JSON is missing build_id: " + targetPath);
        }
        return new TargetSet(schemaVersion, buildId, targets);
    }

    private void readTargetFunctions(
            JsonReader reader, Path targetPath, List<Target> targets) throws Exception {
        reader.beginArray();
        while (reader.hasNext()) {
            String name = null;
            String metadataName = null;
            String signature = null;
            String rva = null;
            reader.beginObject();
            while (reader.hasNext()) {
                String field = reader.nextName();
                switch (field) {
                    case "name":
                        name = reader.nextString();
                        break;
                    case "metadata_name":
                        metadataName = reader.nextString();
                        break;
                    case "signature":
                        signature = reader.nextString();
                        break;
                    case "rva":
                        rva = reader.nextString();
                        break;
                    default:
                        reader.skipValue();
                        break;
                }
            }
            reader.endObject();
            if (name == null || metadataName == null || signature == null || rva == null) {
                throw new IllegalArgumentException(
                    "Incomplete function target in " + targetPath
                );
            }
            targets.add(new Target(name, metadataName, signature, rva));
        }
        reader.endArray();
    }

    private void writeSummaryAtomically(
            Path summaryPath,
            Path summaryTempPath,
            TargetSet targetSet,
            int applied,
            int validated,
            int importedDataTypes,
            int sevenArgumentTargets,
            int preservedLabels) throws Exception {
        Path parent = summaryPath.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Files.deleteIfExists(summaryTempPath);
        try (BufferedWriter buffered = Files.newBufferedWriter(
                summaryTempPath,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE_NEW,
                StandardOpenOption.WRITE
            ); JsonWriter writer = new JsonWriter(buffered)) {
            writer.setIndent("  ");
            writer.beginObject();
            writer.name("schema_version").value(SUMMARY_SCHEMA_VERSION);
            writer.name("program").value(currentProgram.getName());
            writer.name("target_build_id").value(targetSet.buildId);
            writer.name("requested").value(targetSet.targets.size());
            writer.name("applied").value(applied);
            writer.name("validated").value(validated);
            writer.name("unique_function_definitions").value(targetSet.targets.size());
            writer.name("imported_datatypes").value(importedDataTypes);
            writer.name("seven_argument_targets").value(sevenArgumentTargets);
            writer.name("preserved_labels").value(preservedLabels);
            writer.name("calling_convention").value(WINDOWS_X64_CALLING_CONVENTION);
            writer.name("cancelled").value(false);
            writer.endObject();
            writer.flush();
        }
        try {
            Files.move(
                summaryTempPath,
                summaryPath,
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING
            );
        }
        catch (AtomicMoveNotSupportedException unsupported) {
            Files.move(
                summaryTempPath,
                summaryPath,
                StandardCopyOption.REPLACE_EXISTING
            );
        }
    }

    private static class TargetSet {
        private final int schemaVersion;
        private final String buildId;
        private final List<Target> targets;

        TargetSet(int schemaVersion, String buildId, List<Target> targets) {
            this.schemaVersion = schemaVersion;
            this.buildId = buildId;
            this.targets = targets;
        }
    }

    private static class Target {
        private final String name;
        private final String metadataName;
        private final String signature;
        private final String rvaText;
        private final long rva;

        Target(String name, String metadataName, String signature, String rvaText) {
            this.name = name;
            this.metadataName = metadataName;
            this.signature = signature;
            this.rvaText = rvaText;
            this.rva = Long.decode(rvaText);
            if (rva <= 0) {
                throw new IllegalArgumentException("Target RVA must be positive: " + rvaText);
            }
        }
    }

    private static class PreparedTarget {
        private final Target target;
        private final Address address;
        private final Function function;
        private final String functionNameBefore;
        private final Set<String> labelsBefore;
        private final FunctionDefinition archiveDefinition;
        private final FunctionDefinition applyDefinition;

        PreparedTarget(
                Target target,
                Address address,
                Function function,
                String functionNameBefore,
                Set<String> labelsBefore,
                FunctionDefinition archiveDefinition,
                FunctionDefinition applyDefinition) {
            this.target = target;
            this.address = address;
            this.function = function;
            this.functionNameBefore = functionNameBefore;
            this.labelsBefore = labelsBefore;
            this.archiveDefinition = archiveDefinition;
            this.applyDefinition = applyDefinition;
        }
    }
}
