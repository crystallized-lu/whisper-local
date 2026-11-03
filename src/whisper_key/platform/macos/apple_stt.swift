// apple_stt.swift
// Long-lived helper for the `apple` transcription backend: runs macOS 26's
// on-device SpeechTranscriber (the model behind system Dictation). Compiled on
// first use by whisper_engine_apple.py. Audio arrives over stdin, never on disk.
//   in:  uint32 LE sample count, then that many float32 LE samples (16 kHz mono)
//   out: one JSON line at startup ({"ready": true}), then one per utterance:
//        {"text": "...", "secs": 0.4} or {"error": "..."}

import AVFoundation
import Foundation
import Speech

// MARK: - I/O helpers

let inputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000,
                                channels: 1, interleaved: false)!

func emit(_ obj: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: obj)
    FileHandle.standardOutput.write(data + Data([0x0A]))
}

// Reads exactly n bytes from stdin; nil on EOF, which means the app closed the pipe.
func readExactly(_ n: Int) -> Data? {
    var buf = Data()
    while buf.count < n {
        guard let chunk = try? FileHandle.standardInput.read(upToCount: n - buf.count),
              !chunk.isEmpty else { return nil }
        buf.append(chunk)
    }
    return buf
}

// The analyzer dictates its own preferred format; resample our 16 kHz float32 into it.
func convert(_ samples: [Float], to format: AVAudioFormat) throws -> AVAudioPCMBuffer {
    let src = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: AVAudioFrameCount(samples.count))!
    src.frameLength = src.frameCapacity
    samples.withUnsafeBufferPointer { src.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
    if format == inputFormat { return src }

    let converter = AVAudioConverter(from: inputFormat, to: format)!
    let capacity = AVAudioFrameCount(Double(samples.count) * format.sampleRate / 16000) + 1024
    let dst = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity)!
    var fed = false
    var error: NSError?
    converter.convert(to: dst, error: &error) { _, status in
        if fed { status.pointee = .endOfStream; return nil }
        fed = true
        status.pointee = .haveData
        return src
    }
    if let error { throw error }
    return dst
}

// MARK: - Transcription

// An analyzer finishes after one utterance, so we keep the next one prepared
// while the user isn't speaking; that hides the model warm-up from the hotkey.
struct Session {
    let transcriber: SpeechTranscriber
    let analyzer: SpeechAnalyzer
    let format: AVAudioFormat
}

func makeSession(_ locale: Locale) async throws -> Session {
    let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    guard let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber]) else {
        throw NSError(domain: "apple_stt", code: 1, userInfo: [NSLocalizedDescriptionKey: "no audio format"])
    }
    try await analyzer.prepareToAnalyze(in: format)
    return Session(transcriber: transcriber, analyzer: analyzer, format: format)
}

func transcribe(_ samples: [Float], with session: Session) async throws -> String {
    if samples.isEmpty { return "" }
    let buffer = try convert(samples, to: session.format)
    let (stream, input) = AsyncStream<AnalyzerInput>.makeStream()
    async let text = session.transcriber.results.reduce(into: "") { $0 += String($1.text.characters) }
    try await session.analyzer.start(inputSequence: stream)
    input.yield(AnalyzerInput(buffer: buffer))
    input.finish()
    try await session.analyzer.finalizeAndFinishThroughEndOfInput()
    return try await text.trimmingCharacters(in: .whitespacesAndNewlines)
}

// MARK: - Main loop

let wanted = Locale(identifier: CommandLine.arguments.dropFirst().first ?? "en-US")
guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: wanted) else {
    emit(["error": "unsupported locale \(wanted.identifier)"])
    exit(2)
}

// One-time model download (Apple's servers, first run per locale only).
let probe = SpeechTranscriber(locale: locale, preset: .transcription)
if let request = try await AssetInventory.assetInstallationRequest(supporting: [probe]) {
    try await request.downloadAndInstall()
}

var next = try await makeSession(locale)
emit(["ready": true, "locale": locale.identifier])

while let header = readExactly(4) {
    let count = Int(UInt32(littleEndian: header.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }))
    guard let body = readExactly(count * 4) else { break }
    let samples = body.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }

    let started = Date()
    do {
        let text = try await transcribe(samples, with: next)
        emit(["text": text, "secs": Date().timeIntervalSince(started)])
    } catch {
        emit(["error": error.localizedDescription])
    }
    next = try await makeSession(locale)
}
