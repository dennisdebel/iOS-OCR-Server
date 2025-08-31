//
//  TranslationBridge.swift
//  OcrServer
//
//  Created by Apple on 8/30/25.
//

import SwiftUI
import Translation

/// Presents no UI; it just vends a TranslationSession to your app.
struct TranslationBridge: View {
    @State private var config: TranslationSession.Configuration?

    let source: Locale.Language?
    let target: Locale.Language
    let onSession: (TranslationSession) -> Void

    var body: some View {
        Color.clear
            .onAppear {
                // Kick off when this view appears (can live hidden in your root VC/Scene).
                config = TranslationSession.Configuration(source: source, target: target)
            }
            .translationTask(config) { session in
                // Optionally pre-download models so the first translation is instant:
                try? await session.prepareTranslation()
                onSession(session)
                // Invalidate so we don't loop:
                await MainActor.run { config?.invalidate() }
            }
    }
}
