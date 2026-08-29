// Decompiles selected RVAs from the analyzed native program into local files.
// @category DemonBluff

import java.io.BufferedReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import com.google.gson.stream.JsonReader;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.symbol.Symbol;
import ghidra.program.model.symbol.SymbolUtilities;

public class ExportFunctionDecompilations extends GhidraScript {
    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length != 2) {
            throw new IllegalArgumentException(
                "Expected two arguments: output directory and target JSON path"
            );
        }

        Path outputDirectory = Path.of(args[0]).toAbsolutePath().normalize();
        Path targetPath = Path.of(args[1]).toAbsolutePath().normalize();
        List<Target> targets = readTargets(targetPath);
        if (targets.isEmpty()) {
            throw new IllegalArgumentException("Target set contains no functions: " + targetPath);
        }
        Files.createDirectories(outputDirectory);
        DecompInterface decompiler = new DecompInterface();
        decompiler.toggleCCode(true);
        decompiler.toggleSyntaxTree(true);
        if (!decompiler.openProgram(currentProgram)) {
            throw new IllegalStateException("Could not open current program in decompiler");
        }

        int exported = 0;
        int failed = 0;
        int processed = 0;
        try {
            for (Target target : targets) {
                if (monitor.isCancelled()) {
                    break;
                }
                processed++;
                long rva = target.rva;
                String requestedName = target.name;
                String fileName = requestedName.replaceAll("[^A-Za-z0-9_.-]", "_");
                Address address = currentProgram.getImageBase().add(rva);
                String expectedSymbol = SymbolUtilities.replaceInvalidChars(
                    target.metadataName,
                    true
                );
                List<String> aliases = new ArrayList<>();
                boolean foundExpectedSymbol = false;
                for (Symbol symbol : currentProgram.getSymbolTable().getSymbols(address)) {
                    aliases.add(symbol.getName());
                    if (symbol.getName().equals(expectedSymbol)) {
                        foundExpectedSymbol = true;
                    }
                }
                if (!foundExpectedSymbol) {
                    printerr(
                        "Expected imported symbol " + expectedSymbol + " is missing at RVA " +
                        target.rvaText
                    );
                    failed++;
                    continue;
                }
                Function function = getFunctionAt(address);
                if (function == null) {
                    printerr("No function at RVA " + target.rvaText);
                    failed++;
                    continue;
                }

                monitor.setMessage("Decompiling " + function.getName());
                DecompileResults result = decompiler.decompileFunction(function, 180, monitor);
                StringBuilder output = new StringBuilder();
                output.append("/*\n");
                output.append(" * Program: ").append(currentProgram.getName()).append("\n");
                output.append(" * RVA: ").append(target.rvaText).append("\n");
                output.append(" * Address: ").append(address).append("\n");
                output.append(" * Requested target: ").append(requestedName).append("\n");
                output.append(" * Metadata name: ").append(target.metadataName).append("\n");
                output.append(" * Metadata signature: ").append(target.signature).append("\n");
                output.append(" * Native aliases at RVA: ").append(aliases.size()).append("\n");
                output.append(" * Ghidra symbols: ").append(String.join(", ", aliases)).append("\n");
                output.append(" * Function: ").append(function.getName(true)).append("\n");
                output.append(" * Completed: ").append(result.decompileCompleted()).append("\n");
                if (!result.decompileCompleted()) {
                    output.append(" * Error: ").append(result.getErrorMessage()).append("\n");
                }
                output.append(" */\n\n");
                if (result.decompileCompleted() && result.getDecompiledFunction() != null) {
                    output.append(result.getDecompiledFunction().getC());
                    exported++;
                }
                else {
                    output.append("/* DECOMPILATION FAILED */\n");
                    failed++;
                }
                Files.writeString(
                    outputDirectory.resolve(fileName + ".c"),
                    output.toString(),
                    StandardCharsets.UTF_8
                );
            }
        }
        finally {
            decompiler.dispose();
        }
        boolean cancelled = monitor.isCancelled();
        String summary = String.format(
            "{\n  \"requested\": %d,\n  \"processed\": %d,\n  \"exported\": %d,\n" +
            "  \"failed\": %d,\n  \"cancelled\": %s\n}\n",
            targets.size(),
            processed,
            exported,
            failed,
            Boolean.toString(cancelled)
        );
        Files.writeString(
            outputDirectory.resolve("_export_summary.json"),
            summary,
            StandardCharsets.UTF_8
        );
        println("Function exports complete: exported=" + exported + ", failed=" + failed);
    }

    private List<Target> readTargets(Path targetPath) throws Exception {
        List<Target> targets = new ArrayList<>();
        try (BufferedReader buffered = Files.newBufferedReader(targetPath, StandardCharsets.UTF_8);
                JsonReader reader = new JsonReader(buffered)) {
            reader.beginObject();
            while (reader.hasNext()) {
                String section = reader.nextName();
                if (!section.equals("functions")) {
                    reader.skipValue();
                    continue;
                }
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
                        throw new IllegalArgumentException("Incomplete function target in " + targetPath);
                    }
                    targets.add(new Target(name, metadataName, signature, rva));
                }
                reader.endArray();
            }
            reader.endObject();
        }
        return targets;
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
        }
    }
}
