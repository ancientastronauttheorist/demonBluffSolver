// Imports Il2CppDumper script.json labels before Ghidra auto-analysis.
// @category DemonBluff

import java.io.BufferedReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;

import com.google.gson.stream.JsonReader;

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.symbol.SourceType;
import ghidra.program.model.symbol.SymbolUtilities;

public class ImportIl2CppSymbols extends GhidraScript {
    private final Set<Long> functionAddresses = new HashSet<>();
    private long methodLabels;
    private long metadataLabels;
    private long stringLabels;
    private long skippedAddresses;

    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length != 1) {
            throw new IllegalArgumentException("Expected one argument: path to Il2CppDumper script.json");
        }

        Path jsonPath = Path.of(args[0]).toAbsolutePath().normalize();
        println("Importing IL2CPP symbols from " + jsonPath);
        try (BufferedReader buffered = Files.newBufferedReader(jsonPath, StandardCharsets.UTF_8);
                JsonReader reader = new JsonReader(buffered)) {
            reader.beginObject();
            while (reader.hasNext() && !monitor.isCancelled()) {
                String section = reader.nextName();
                switch (section) {
                    case "ScriptMethod":
                        readMethods(reader);
                        break;
                    case "ScriptString":
                        readStrings(reader);
                        break;
                    case "ScriptMetadata":
                        readMetadata(reader);
                        break;
                    case "ScriptMetadataMethod":
                        readMetadataMethods(reader);
                        break;
                    default:
                        reader.skipValue();
                        break;
                }
            }
            reader.endObject();
        }

        println("IL2CPP symbol import complete:");
        println("  method labels:   " + methodLabels);
        println("  unique functions:" + functionAddresses.size());
        println("  metadata labels: " + metadataLabels);
        println("  string labels:   " + stringLabels);
        println("  skipped RVAs:    " + skippedAddresses);
    }

    private void readMethods(JsonReader reader) throws Exception {
        reader.beginArray();
        while (reader.hasNext() && !monitor.isCancelled()) {
            long rva = -1;
            String name = null;
            reader.beginObject();
            while (reader.hasNext()) {
                String field = reader.nextName();
                if (field.equals("Address")) {
                    rva = reader.nextLong();
                }
                else if (field.equals("Name")) {
                    name = reader.nextString();
                }
                else {
                    reader.skipValue();
                }
            }
            reader.endObject();
            Address address = addressForRva(rva);
            if (address == null || name == null) {
                continue;
            }
            String safeName = safeName(name);
            addLabel(address, safeName);
            methodLabels++;
            if (functionAddresses.add(rva)) {
                Function function = getFunctionAt(address);
                if (function == null) {
                    try {
                        createFunction(address, safeName);
                    }
                    catch (Exception exception) {
                        printerr("Could not create function at " + address + ": " + exception.getMessage());
                    }
                }
            }
        }
        reader.endArray();
    }

    private void readStrings(JsonReader reader) throws Exception {
        long index = 1;
        reader.beginArray();
        while (reader.hasNext() && !monitor.isCancelled()) {
            long rva = -1;
            String value = null;
            reader.beginObject();
            while (reader.hasNext()) {
                String field = reader.nextName();
                if (field.equals("Address")) {
                    rva = reader.nextLong();
                }
                else if (field.equals("Value")) {
                    value = reader.nextString();
                }
                else {
                    reader.skipValue();
                }
            }
            reader.endObject();
            Address address = addressForRva(rva);
            if (address != null) {
                addLabel(address, "StringLiteral_" + index);
                if (value != null) {
                    setEOLComment(address, value);
                }
                stringLabels++;
            }
            index++;
        }
        reader.endArray();
    }

    private void readMetadata(JsonReader reader) throws Exception {
        reader.beginArray();
        while (reader.hasNext() && !monitor.isCancelled()) {
            long rva = -1;
            String name = null;
            reader.beginObject();
            while (reader.hasNext()) {
                String field = reader.nextName();
                if (field.equals("Address")) {
                    rva = reader.nextLong();
                }
                else if (field.equals("Name")) {
                    name = reader.nextString();
                }
                else {
                    reader.skipValue();
                }
            }
            reader.endObject();
            Address address = addressForRva(rva);
            if (address != null && name != null) {
                addLabel(address, safeName(name));
                setEOLComment(address, name);
                metadataLabels++;
            }
        }
        reader.endArray();
    }

    private void readMetadataMethods(JsonReader reader) throws Exception {
        reader.beginArray();
        while (reader.hasNext() && !monitor.isCancelled()) {
            long rva = -1;
            String name = null;
            reader.beginObject();
            while (reader.hasNext()) {
                String field = reader.nextName();
                if (field.equals("Address")) {
                    rva = reader.nextLong();
                }
                else if (field.equals("Name")) {
                    name = reader.nextString();
                }
                else {
                    reader.skipValue();
                }
            }
            reader.endObject();
            Address address = addressForRva(rva);
            if (address != null && name != null) {
                addLabel(address, safeName(name));
                setEOLComment(address, name);
                metadataLabels++;
            }
        }
        reader.endArray();
    }

    private Address addressForRva(long rva) {
        if (rva <= 0) {
            skippedAddresses++;
            return null;
        }
        Address address;
        try {
            address = currentProgram.getImageBase().add(rva);
        }
        catch (Exception exception) {
            skippedAddresses++;
            return null;
        }
        if (!currentProgram.getMemory().contains(address)) {
            skippedAddresses++;
            return null;
        }
        return address;
    }

    private void addLabel(Address address, String name) {
        try {
            createLabel(address, name, false, SourceType.IMPORTED);
        }
        catch (Exception exception) {
            // Duplicate aliases and invalid corner cases are nonfatal; the first
            // valid symbol still gives analysis a useful anchor.
        }
    }

    private String safeName(String value) {
        String result = SymbolUtilities.replaceInvalidChars(value, true);
        if (result.length() > 512) {
            result = result.substring(0, 512);
        }
        return result;
    }
}
