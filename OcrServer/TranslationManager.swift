//
//  TranslationManager.swift
//  OcrServer
//
//  Created by Apple on 8/30/25.
//


import Foundation
import Translation

@MainActor
final class TranslationManager {
    private var session: TranslationSession?

    func attach(session: TranslationSession) { self.session = session }
    

    /// Translate many strings at once and return results in the same order.
    func translateBatch(_ texts: [String]) async throws -> [String] {
        guard let session else { throw NSError(domain: "NoSession", code: 1) }
        let reqs = texts.enumerated().map {
            TranslationSession.Request(sourceText: $0.element, clientIdentifier: "\($0.offset)")
        }
        let responses = try await session.translations(from: reqs)
        // `translations(from:)` preserves order; map to target text.
        return responses.map { $0.targetText }
    }

    /// Stream translations as they’re ready (useful for very large batches).
    func translateStreaming(_ texts: [String], onEach: @escaping (_ index: Int, _ zh: String) -> Void) async throws {
        guard let session else { throw NSError(domain: "NoSession", code: 1) }
        let reqs = texts.enumerated().map {
            TranslationSession.Request(sourceText: $0.element, clientIdentifier: "\($0.offset)")
        }
        for try await response in session.translate(batch: reqs) {
            if let id = response.clientIdentifier, let idx = Int(id) {
                onEach(idx, response.targetText)
            }
        }
    }
}
