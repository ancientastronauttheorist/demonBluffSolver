// Records whether a headless analysis pass reached its post-script without timing out.
// @category DemonBluff

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import ghidra.app.util.headless.HeadlessScript;

public class RecordAnalysisCompletion extends HeadlessScript {
    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length != 1) {
            throw new IllegalArgumentException("Expected completion-summary path");
        }
        Path summaryPath = Path.of(args[0]).toAbsolutePath().normalize();
        boolean timedOut = analysisTimeoutOccurred();
        String programName = currentProgram.getName().replace("\\", "\\\\").replace("\"", "\\\"");
        String summary = String.format(
            "{\n  \"program\": \"%s\",\n  \"analysis_timeout_occurred\": %s\n}\n",
            programName,
            Boolean.toString(timedOut)
        );
        Files.createDirectories(summaryPath.getParent());
        Files.writeString(summaryPath, summary, StandardCharsets.UTF_8);
        println("Analysis completion recorded: timeout=" + timedOut);
    }
}
