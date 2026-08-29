// Builds a validated, build-keyed Ghidra datatype archive from normalized IL2CPP headers.
// @category DemonBluff

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.UUID;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.google.gson.stream.JsonReader;

import ghidra.app.util.cparser.C.CParserUtils;
import ghidra.app.util.headless.HeadlessScript;
import ghidra.program.model.data.Composite;
import ghidra.program.model.data.DataType;
import ghidra.program.model.data.DataTypeManager;
import ghidra.program.model.data.FileDataTypeManager;
import ghidra.program.model.data.FunctionDefinition;
import ghidra.program.model.data.ParameterDefinition;
import ghidra.program.model.data.Pointer;
import ghidra.program.model.data.Structure;

public class BuildIl2CppTypeArchive extends HeadlessScript {
    private static final String LANGUAGE_ID = "x86:LE:64:default";
    private static final String COMPILER_SPEC_ID = "windows";
    private static final Pattern BUILD_ID_PATTERN =
        Pattern.compile("[0-9a-f]{12}_[0-9a-f]{12}");
    private static final Pattern FUNCTION_NAME_PATTERN =
        Pattern.compile("\\b([A-Za-z_][A-Za-z0-9_]*)\\s*\\(");

    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 6) {
            throw new IllegalArgumentException(
                "Expected at least six arguments: normalized-header, prototype-header, " +
                "alignment-manifest, output-directory, success-summary, target-json [...]"
            );
        }

        Path normalizedHeader = requireFile(args[0], "normalized header");
        Path prototypeHeader = requireFile(args[1], "prototype header");
        Path alignmentManifestPath = requireFile(args[2], "alignment manifest");
        Path outputDirectory = Path.of(args[3]).toAbsolutePath().normalize();
        Path summaryPath = Path.of(args[4]).toAbsolutePath().normalize();
        List<Path> targetPaths = new ArrayList<>();
        for (int index = 5; index < args.length; index++) {
            targetPaths.add(requireFile(args[index], "target JSON"));
        }
        Files.createDirectories(outputDirectory);
        if (summaryPath.getParent() != null) {
            Files.createDirectories(summaryPath.getParent());
        }
        Files.deleteIfExists(summaryPath);

        AlignmentManifest alignmentManifest = readAlignmentManifest(alignmentManifestPath);
        TargetSet targetSet = readAndMergeTargets(targetPaths);
        requireEqual("alignment/target build_id", alignmentManifest.buildId, targetSet.buildId);
        if (!BUILD_ID_PATTERN.matcher(targetSet.buildId).matches()) {
            throw new IllegalArgumentException("Invalid build_id: " + targetSet.buildId);
        }
        requireEqual(
            "normalized-header SHA-256",
            alignmentManifest.normalizedHeaderSha256,
            sha256(normalizedHeader)
        );
        requireEqual(
            "prototype-header SHA-256",
            alignmentManifest.prototypeHeaderSha256,
            sha256(prototypeHeader)
        );
        if (alignmentManifest.alignment != 8) {
            throw new IllegalArgumentException(
                "Only explicit align(8) is supported, found " + alignmentManifest.alignment
            );
        }
        if (alignmentManifest.names.size() != alignmentManifest.alignmentCount) {
            throw new IllegalArgumentException(
                "Alignment count does not match name list: " +
                alignmentManifest.alignmentCount + " != " + alignmentManifest.names.size()
            );
        }
        if (new HashSet<>(alignmentManifest.names).size() != alignmentManifest.names.size()) {
            throw new IllegalArgumentException("Alignment manifest contains duplicate names");
        }
        if (!alignmentManifest.prototypeNames.equals(targetSet.prototypeNames())) {
            throw new IllegalArgumentException(
                "Alignment manifest prototype list does not match target JSON"
            );
        }

        Path finalArchive = outputDirectory.resolve(
            "il2cpp-types-" + targetSet.buildId + ".gdt"
        );
        Path stagingDirectory = outputDirectory.resolve(".staging-" + UUID.randomUUID());
        Files.createDirectory(stagingDirectory);
        Path partialRequested = stagingDirectory.resolve(finalArchive.getFileName());
        Path actualPartial = partialRequested;
        Path parserOutput = Path.of(partialRequested.toString() + "_CParser.out");
        FileDataTypeManager dataTypes = null;
        boolean promoted = false;
        int dataTypeCount = 0;
        int functionDefinitionCount = 0;

        try {
            dataTypes = FileDataTypeManager.createFileArchive(
                partialRequested.toFile(), LANGUAGE_ID, COMPILER_SPEC_ID
            );
            actualPartial = Path.of(dataTypes.getPath()).toAbsolutePath().normalize();
            parserOutput = Path.of(actualPartial.toString() + "_CParser.out");

            CParserUtils.CParseResults parseResults = CParserUtils.parseHeaderFiles(
                new DataTypeManager[0],
                new String[] {
                    normalizedHeader.toString(),
                    prototypeHeader.toString()
                },
                new String[0],
                new String[] { "-v0" },
                dataTypes,
                monitor
            );
            if (monitor.isCancelled()) {
                throw new IllegalStateException("Datatype archive build was cancelled");
            }
            if (!parseResults.successful()) {
                throw new IllegalStateException(
                    parseResults.getFormattedParseMessage("C parser did not complete")
                );
            }
            if (!parseResults.cppParseMessages().isBlank() ||
                !parseResults.cParseMessages().isBlank()) {
                throw new IllegalStateException(
                    parseResults.getFormattedParseMessage("C parser emitted diagnostics")
                );
            }

            restoreAlignments(dataTypes, alignmentManifest.names);
            validateCriticalLayouts(dataTypes);
            functionDefinitionCount = validateFunctionDefinitions(dataTypes, targetSet.targets);
            dataTypeCount = dataTypes.getDataTypeCount(true);
            dataTypes.save();
            dataTypes.close();
            dataTypes = null;

            atomicReplace(actualPartial, finalArchive);
            Path finalParserOutput = Path.of(finalArchive.toString() + "_CParser.out");
            if (Files.exists(parserOutput)) {
                atomicReplace(parserOutput, finalParserOutput);
            }

            String summary = String.format(
                "{\n" +
                "  \"alignment_count\": %d,\n" +
                "  \"build_id\": \"%s\",\n" +
                "  \"critical_layouts_validated\": true,\n" +
                "  \"data_type_count\": %d,\n" +
                "  \"function_definition_count\": %d,\n" +
                "  \"gdt_path\": \"%s\",\n" +
                "  \"normalized_header_sha256\": \"%s\",\n" +
                "  \"prototype_header_sha256\": \"%s\",\n" +
                "  \"schema_version\": 1,\n" +
                "  \"success\": true\n" +
                "}\n",
                alignmentManifest.alignmentCount,
                jsonEscape(targetSet.buildId),
                dataTypeCount,
                functionDefinitionCount,
                jsonEscape(finalArchive.toString()),
                alignmentManifest.normalizedHeaderSha256,
                alignmentManifest.prototypeHeaderSha256
            );
            atomicWrite(summaryPath, summary);
            promoted = true;
            println("IL2CPP datatype archive built: " + finalArchive);
            println("Validated function definitions: " + functionDefinitionCount);
        }
        catch (Throwable failure) {
            setHeadlessContinuationOption(HeadlessContinuationOption.ABORT);
            throw failure;
        }
        finally {
            if (dataTypes != null) {
                dataTypes.close();
            }
            if (!promoted) {
                Files.deleteIfExists(actualPartial);
                Files.deleteIfExists(parserOutput);
                Files.deleteIfExists(Path.of(actualPartial.toString() + ".ulock"));
            }
            Files.deleteIfExists(stagingDirectory);
        }
    }

    private void restoreAlignments(
            FileDataTypeManager dataTypes, List<String> alignedNames) throws Exception {
        int transaction = dataTypes.startTransaction("Restore IL2CPP align(8)");
        boolean commit = false;
        try {
            for (String name : alignedNames) {
                Structure structure = requireUniqueStructure(dataTypes, name);
                structure.setExplicitMinimumAlignment(8);
                structure.repack();
            }
            List<Composite> composites = new ArrayList<>();
            Iterator<Composite> iterator = dataTypes.getAllComposites();
            while (iterator.hasNext()) {
                composites.add(iterator.next());
            }
            for (Composite composite : composites) {
                composite.repack();
            }
            for (String name : alignedNames) {
                Structure structure = requireUniqueStructure(dataTypes, name);
                if (!structure.hasExplicitMinimumAlignment() ||
                    structure.getExplicitMinimumAlignment() != 8) {
                    throw new IllegalStateException("Alignment was not restored for " + name);
                }
            }
            commit = true;
        }
        finally {
            dataTypes.endTransaction(transaction, commit);
        }
    }

    private void validateCriticalLayouts(DataTypeManager dataTypes) {
        requireLayout(dataTypes, "MethodInfo", 0x58,
            new Field("token", 0x48), new Field("parameters_count", 0x52));
        requireLayout(dataTypes, "CharactersCount_Fields", 0x28,
            new Field("allCharCount", 0x0), new Field("dMinion", 0x20));
        requireLayout(dataTypes, "CharactersCount_o", 0x38,
            new Field("klass", 0x0), new Field("monitor", 0x8), new Field("fields", 0x10));
        requireLayout(dataTypes, "Gameplay_Fields", 0x80,
            new Field("super", 0x0), new Field("roguelikeDeck", 0x10),
            new Field("specialRules", 0x78));
        requireLayout(dataTypes, "Gameplay_o", 0x90, new Field("fields", 0x10));
        requireLayout(dataTypes, "PlayerController_Fields", 0x28,
            new Field("super", 0x0), new Field("modeChange", 0x20));
        requireLayout(dataTypes, "PlayerController_o", 0x38, new Field("fields", 0x10));
        requireLayout(dataTypes, "Health_Fields", 0x8, new Field("super", 0x0));
        requireLayout(dataTypes, "Health_o", 0x18, new Field("fields", 0x10));
        requireLayout(dataTypes, "CurrentMaxValue_Fields", 0x10,
            new Field("super", 0x0), new Field("max", 0x8), new Field("current", 0xc));
        requireLayout(dataTypes, "CurrentMaxValue_o", 0x20, new Field("fields", 0x10));
        requireLayout(dataTypes, "System_Collections_Generic_List_CharacterData__Fields", 0x18,
            new Field("_items", 0x0), new Field("_syncRoot", 0x10));
        requireLayout(dataTypes, "System_Collections_Generic_List_CharacterData__o", 0x28,
            new Field("fields", 0x10));
    }

    private int validateFunctionDefinitions(DataTypeManager dataTypes, List<Target> targets) {
        int validated = 0;
        for (Target target : targets) {
            List<DataType> matches = new ArrayList<>();
            dataTypes.findDataTypes(target.prototypeName, matches);
            List<FunctionDefinition> definitions = new ArrayList<>();
            for (DataType match : matches) {
                if (match.getName().equals(target.prototypeName) &&
                    match instanceof FunctionDefinition definition) {
                    definitions.add(definition);
                }
            }
            if (definitions.size() != 1) {
                throw new IllegalStateException(
                    "Expected one function definition named " + target.prototypeName +
                    ", found " + definitions.size()
                );
            }
            FunctionDefinition definition = definitions.get(0);
            ParameterDefinition[] actualArguments = definition.getArguments();
            if (actualArguments.length != target.parameters.size()) {
                throw new IllegalStateException(
                    target.prototypeName + " has " + actualArguments.length +
                    " parameters; expected " + target.parameters.size()
                );
            }
            requireEqual(
                target.prototypeName + " return type",
                target.returnType,
                dataTypeKey(definition.getReturnType())
            );
            for (int index = 0; index < actualArguments.length; index++) {
                ParameterExpectation expected = target.parameters.get(index);
                ParameterDefinition actual = actualArguments[index];
                requireEqual(
                    target.prototypeName + " parameter " + index + " name",
                    expected.name,
                    actual.getName()
                );
                requireEqual(
                    target.prototypeName + " parameter " + expected.name + " type",
                    expected.type,
                    dataTypeKey(actual.getDataType())
                );
            }
            if (definition.hasVarArgs()) {
                throw new IllegalStateException(
                    "Unexpected variadic function definition: " + target.prototypeName
                );
            }
            validated++;
        }
        return validated;
    }

    private void requireLayout(
            DataTypeManager dataTypes, String name, int length, Field... fields) {
        Structure structure = requireUniqueStructure(dataTypes, name);
        if (structure.getLength() != length) {
            throw new IllegalStateException(
                name + " has length 0x" + Integer.toHexString(structure.getLength()) +
                "; expected 0x" + Integer.toHexString(length)
            );
        }
        for (Field field : fields) {
            var component = structure.findComponent(field.name);
            if (component == null || component.getOffset() != field.offset) {
                String actual = component == null
                    ? "missing"
                    : "0x" + Integer.toHexString(component.getOffset());
                throw new IllegalStateException(
                    name + "." + field.name + " is " + actual +
                    "; expected 0x" + Integer.toHexString(field.offset)
                );
            }
        }
    }

    private Structure requireUniqueStructure(DataTypeManager dataTypes, String name) {
        List<DataType> matches = new ArrayList<>();
        dataTypes.findDataTypes(name, matches);
        List<Structure> structures = new ArrayList<>();
        for (DataType match : matches) {
            if (match.getName().equals(name) && match instanceof Structure structure) {
                structures.add(structure);
            }
        }
        if (structures.size() != 1) {
            throw new IllegalStateException(
                "Expected one structure named " + name + ", found " + structures.size()
            );
        }
        return structures.get(0);
    }

    private AlignmentManifest readAlignmentManifest(Path path) throws Exception {
        String buildId = null;
        String normalizedHash = null;
        String prototypeHash = null;
        int alignment = -1;
        int alignmentCount = -1;
        List<String> names = new ArrayList<>();
        List<String> prototypeNames = new ArrayList<>();
        try (BufferedReader buffered = Files.newBufferedReader(path, StandardCharsets.UTF_8);
                JsonReader reader = new JsonReader(buffered)) {
            reader.beginObject();
            while (reader.hasNext()) {
                String field = reader.nextName();
                switch (field) {
                    case "alignment": alignment = reader.nextInt(); break;
                    case "alignment_count": alignmentCount = reader.nextInt(); break;
                    case "build_id": buildId = reader.nextString(); break;
                    case "names": readStrings(reader, names); break;
                    case "prototype_names": readStrings(reader, prototypeNames); break;
                    case "outputs":
                        reader.beginObject();
                        while (reader.hasNext()) {
                            String output = reader.nextName();
                            if (output.equals("normalized_header_sha256")) {
                                normalizedHash = reader.nextString();
                            }
                            else if (output.equals("prototype_header_sha256")) {
                                prototypeHash = reader.nextString();
                            }
                            else {
                                reader.skipValue();
                            }
                        }
                        reader.endObject();
                        break;
                    default: reader.skipValue();
                }
            }
            reader.endObject();
        }
        if (buildId == null || normalizedHash == null || prototypeHash == null ||
            alignment < 0 || alignmentCount < 0) {
            throw new IllegalArgumentException("Incomplete alignment manifest: " + path);
        }
        return new AlignmentManifest(
            buildId, alignment, alignmentCount, names, prototypeNames,
            normalizedHash.toUpperCase(), prototypeHash.toUpperCase()
        );
    }

    private TargetSet readAndMergeTargets(List<Path> paths) throws Exception {
        String buildId = null;
        Map<String, Target> byName = new TreeMap<>();
        for (Path path : paths) {
            TargetSet targetSet = readTargets(path);
            if (buildId == null) {
                buildId = targetSet.buildId;
            }
            else {
                requireEqual("target build_id", buildId, targetSet.buildId);
            }
            for (Target target : targetSet.targets) {
                Target previous = byName.putIfAbsent(target.prototypeName, target);
                if (previous != null && !previous.signature.equals(target.signature)) {
                    throw new IllegalArgumentException(
                        "Conflicting signatures for " + target.prototypeName + ": " +
                        previous.signature + " != " + target.signature
                    );
                }
            }
        }
        if (buildId == null || byName.isEmpty()) {
            throw new IllegalArgumentException("No target functions were supplied");
        }
        return new TargetSet(buildId, new ArrayList<>(byName.values()));
    }

    private TargetSet readTargets(Path path) throws Exception {
        String buildId = null;
        List<Target> targets = new ArrayList<>();
        try (BufferedReader buffered = Files.newBufferedReader(path, StandardCharsets.UTF_8);
                JsonReader reader = new JsonReader(buffered)) {
            reader.beginObject();
            while (reader.hasNext()) {
                String field = reader.nextName();
                if (field.equals("build_id")) {
                    buildId = reader.nextString();
                }
                else if (field.equals("functions")) {
                    reader.beginArray();
                    while (reader.hasNext()) {
                        String signature = null;
                        reader.beginObject();
                        while (reader.hasNext()) {
                            String targetField = reader.nextName();
                            if (targetField.equals("signature")) {
                                signature = reader.nextString();
                            }
                            else {
                                reader.skipValue();
                            }
                        }
                        reader.endObject();
                        if (signature == null) {
                            throw new IllegalArgumentException("Target has no signature: " + path);
                        }
                        targets.add(parseTarget(signature));
                    }
                    reader.endArray();
                }
                else {
                    reader.skipValue();
                }
            }
            reader.endObject();
        }
        if (buildId == null || targets.isEmpty()) {
            throw new IllegalArgumentException("Incomplete target JSON: " + path);
        }
        return new TargetSet(buildId, targets);
    }

    private Target parseTarget(String signature) {
        Matcher matcher = FUNCTION_NAME_PATTERN.matcher(signature);
        if (!matcher.find()) {
            throw new IllegalArgumentException("Cannot identify function in: " + signature);
        }
        String name = matcher.group(1);
        String returnType = sourceTypeKey(signature.substring(0, matcher.start(1)));
        int open = signature.indexOf('(', matcher.start(1) + name.length());
        int close = matchingParenthesis(signature, open);
        String parameters = signature.substring(open + 1, close).trim();
        List<ParameterExpectation> expectations = parseParameters(parameters, signature);
        return new Target(name, signature.trim(), returnType, expectations);
    }

    private int matchingParenthesis(String value, int open) {
        int depth = 0;
        for (int index = open; index < value.length(); index++) {
            char character = value.charAt(index);
            if (character == '(') depth++;
            else if (character == ')' && --depth == 0) return index;
        }
        throw new IllegalArgumentException("Unbalanced signature: " + value);
    }

    private List<ParameterExpectation> parseParameters(
            String parameters, String completeSignature) {
        List<ParameterExpectation> expectations = new ArrayList<>();
        if (parameters.isEmpty() || parameters.equals("void")) return expectations;
        List<String> declarations = new ArrayList<>();
        int depth = 0;
        int start = 0;
        for (int index = 0; index < parameters.length(); index++) {
            char character = parameters.charAt(index);
            if (character == '(' || character == '[') depth++;
            else if (character == ')' || character == ']') depth--;
            else if (character == ',' && depth == 0) {
                declarations.add(parameters.substring(start, index).trim());
                start = index + 1;
            }
        }
        declarations.add(parameters.substring(start).trim());
        Pattern parameterNamePattern = Pattern.compile("([A-Za-z_][A-Za-z0-9_]*)\\s*$");
        for (String declaration : declarations) {
            Matcher nameMatcher = parameterNamePattern.matcher(declaration);
            if (!nameMatcher.find()) {
                throw new IllegalArgumentException(
                    "Cannot identify parameter in signature: " + completeSignature
                );
            }
            String parameterName = nameMatcher.group(1);
            String parameterType = sourceTypeKey(
                declaration.substring(0, nameMatcher.start(1))
            );
            expectations.add(new ParameterExpectation(parameterName, parameterType));
        }
        return expectations;
    }

    private String sourceTypeKey(String source) {
        return source
            .replaceAll("\\b(?:const|volatile|struct)\\b", "")
            .replaceAll("\\s+", "")
            .trim();
    }

    private String dataTypeKey(DataType dataType) {
        if (dataType instanceof Pointer pointer) {
            return dataTypeKey(pointer.getDataType()) + "*";
        }
        return dataType.getName().replaceAll("\\s+", "");
    }

    private void readStrings(JsonReader reader, List<String> destination) throws IOException {
        reader.beginArray();
        while (reader.hasNext()) destination.add(reader.nextString());
        reader.endArray();
    }

    private Path requireFile(String value, String description) {
        Path path = Path.of(value).toAbsolutePath().normalize();
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException("Missing " + description + ": " + path);
        }
        return path;
    }

    private String sha256(Path path) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (var input = Files.newInputStream(path)) {
            byte[] buffer = new byte[1024 * 1024];
            int count;
            while ((count = input.read(buffer)) >= 0) {
                if (count > 0) digest.update(buffer, 0, count);
            }
        }
        StringBuilder hex = new StringBuilder();
        for (byte value : digest.digest()) hex.append(String.format("%02X", value));
        return hex.toString();
    }

    private void atomicReplace(Path source, Path destination) throws IOException {
        try {
            Files.move(
                source, destination,
                StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING
            );
        }
        catch (AtomicMoveNotSupportedException unsupported) {
            Files.move(source, destination, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private void atomicWrite(Path destination, String value) throws IOException {
        Path temporary = destination.resolveSibling(
            "." + destination.getFileName() + "." + UUID.randomUUID() + ".tmp"
        );
        try {
            Files.writeString(temporary, value, StandardCharsets.UTF_8);
            atomicReplace(temporary, destination);
        }
        finally {
            Files.deleteIfExists(temporary);
        }
    }

    private void requireEqual(String description, String expected, String actual) {
        if (!expected.equals(actual)) {
            throw new IllegalArgumentException(
                description + " mismatch: expected " + expected + ", found " + actual
            );
        }
    }

    private String jsonEscape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private record Field(String name, int offset) {}

    private record ParameterExpectation(String name, String type) {}

    private record Target(
        String prototypeName,
        String signature,
        String returnType,
        List<ParameterExpectation> parameters
    ) {}

    private record TargetSet(String buildId, List<Target> targets) {
        List<String> prototypeNames() {
            List<String> names = new ArrayList<>();
            for (Target target : targets) names.add(target.prototypeName);
            return names;
        }
    }

    private record AlignmentManifest(
        String buildId,
        int alignment,
        int alignmentCount,
        List<String> names,
        List<String> prototypeNames,
        String normalizedHeaderSha256,
        String prototypeHeaderSha256
    ) {}
}
