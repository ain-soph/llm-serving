import Foundation
import MLX
import MLXIndexTTS2
import MLXRandom

struct SynthesisRequest: Decodable {
    let reference16k: Data
    let reference22k: Data
    let pieces: [String]
    let language: String?
    let emotion: [Float]?
    let speechRate: Double
    let temperature: Float?
    let seed: UInt64?
}

struct WorkerEvent: Encodable {
    let kind: String
    var samples: Data? = nil
    var message: String? = nil
    var peakBytes: Int? = nil
    var seconds: Double? = nil
}

enum WorkerError: Error {
    case invalidPCM
    case unknownLanguage
    case invalidReference(String)
    case invalidOutput(Int)
}

let decoder = JSONDecoder()
let encoder = JSONEncoder()

func send(_ event: WorkerEvent) throws {
    var wire = try encoder.encode(event)
    wire.append(0x0A)
    try FileHandle.standardOutput.write(contentsOf: wire)
}

func samples(_ bytes: Data) throws -> [Float] {
    let width = MemoryLayout<Float>.size
    guard !bytes.isEmpty, bytes.count.isMultiple(of: width) else {
        throw WorkerError.invalidPCM
    }
    return bytes.withUnsafeBytes { buffer in
        stride(from: 0, to: buffer.count, by: width).map { offset in
            Float(bitPattern: UInt32(littleEndian: buffer.loadUnaligned(fromByteOffset: offset, as: UInt32.self)))
        }
    }
}

func synthesize(_ request: SynthesisRequest, using model: IndexTTS2Generator) throws {
    let language = request.language.flatMap(IndexTTSLanguage.init(parsing:))
    if request.language != nil && language == nil {
        throw WorkerError.unknownLanguage
    }
    let started = ProcessInfo.processInfo.systemUptime
    Memory.peakMemory = 0
    let reference = try model.prepareReference(
        samples16k: samples(request.reference16k), samples22k: samples(request.reference22k))
    for (label, value) in [("style", reference.style), ("emotion", reference.baseEmovec),
                           ("prompt", reference.promptCondition), ("mel", reference.refMel)] {
        if !MLX.all(MLX.isFinite(value)).item(Bool.self) {
            throw WorkerError.invalidReference(label)
        }
    }
    if let seed = request.seed {
        MLXRandom.seed(seed)
    }
    for piece in request.pieces {
        var parameters = IndexTTS2Generator.SynthesisParams()
        parameters.temperature = request.temperature ?? parameters.temperature
        // Acoustic codes are at 25 Hz. A length-dependent cap bounds a run that misses EOS.
        parameters.maxMelTokens = min(700, 150 + 5 * piece.count)
        let audio = try model.synthesize(
            text: piece, reference: reference, language: language, emotionWeights: request.emotion,
            speechRate: request.speechRate, params: parameters)

        guard !audio.isEmpty, audio.allSatisfy(\.isFinite) else {
            throw WorkerError.invalidOutput(audio.count)
        }
        let bytes = audio.withUnsafeBufferPointer { Data(buffer: $0) }
        try send(WorkerEvent(kind: "audio", samples: bytes))
    }
    try send(WorkerEvent(kind: "done", peakBytes: Memory.peakMemory,
                         seconds: ProcessInfo.processInfo.systemUptime - started))
}

func run() throws {
    let arguments = CommandLine.arguments
    guard arguments.count == 3, let bits = Int(arguments[2]), bits == 0 || bits == 8 else {
        FileHandle.standardError.write(Data("Usage: indextts-worker MODEL_DIR 0|8\n".utf8))
        exit(2)
    }
    Memory.cacheLimit = 512 * 1024 * 1024
    let directory = URL(fileURLWithPath: arguments[1], isDirectory: true)
    let started = ProcessInfo.processInfo.systemUptime
    let model = try IndexTTS2Generator.load(
        modelDirectory: directory, w2vBertDirectory: directory, quantBits: bits == 0 ? nil : bits)
    try send(WorkerEvent(kind: "ready", seconds: ProcessInfo.processInfo.systemUptime - started))
    while let line = readLine() {
        do {
            let request = try decoder.decode(SynthesisRequest.self, from: Data(line.utf8))
            try synthesize(request, using: model)
        } catch {
            try send(WorkerEvent(kind: "error", message: String(describing: error)))
        }
        Memory.clearCache()
    }
}

try run()
