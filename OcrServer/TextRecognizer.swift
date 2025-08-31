//
//  TextRecognizer.swift
//  OcrServer
//
//  Created by Riddle Ling on 2025/8/1.
//

import Foundation
import Vision


class TextRecognizer {
    let recognitionLevel : RecognizeTextRequest.RecognitionLevel
    let usesLanguageCorrection : Bool
    let automaticallyDetectsLanguage : Bool
    
    init(recognitionLevel: RecognizeTextRequest.RecognitionLevel = .accurate,
         usesLanguageCorrection: Bool = true,
         automaticallyDetectsLanguage: Bool = true) {
        self.recognitionLevel = recognitionLevel
        self.usesLanguageCorrection = usesLanguageCorrection
        self.automaticallyDetectsLanguage = automaticallyDetectsLanguage
    }
    
    func getOcrResult(data: Data) async -> OCRResult? {
        // 嘗試取影像像素大小
        guard let (W, H) = Self.imagePixelSize(from: data) else {
            return nil
        }
        
        var request = RecognizeTextRequest()
        request.recognitionLevel = recognitionLevel
        request.usesLanguageCorrection = usesLanguageCorrection
        request.automaticallyDetectsLanguage = automaticallyDetectsLanguage
        
        let observations = (try? await request.perform(on: data)) ?? []
        
        var lines: [String] = []
        var items: [OCRBoxItem] = []
        
        for obs in observations {
            guard let best = obs.topCandidates(1).first else { continue }
            let text = best.string
            lines.append(text)
            
            // 四個角轉成像素座標（左上為原點）
            let corners = [
                CGPoint(x: obs.topLeft.x     * CGFloat(W), y: (1 - obs.topLeft.y)     * CGFloat(H)),
                CGPoint(x: obs.topRight.x    * CGFloat(W), y: (1 - obs.topRight.y)    * CGFloat(H)),
                CGPoint(x: obs.bottomRight.x * CGFloat(W), y: (1 - obs.bottomRight.y) * CGFloat(H)),
                CGPoint(x: obs.bottomLeft.x  * CGFloat(W), y: (1 - obs.bottomLeft.y)  * CGFloat(H))
            ]
            
            // 取最小外接矩形
            let minX = corners.map { $0.x }.min() ?? 0
            let maxX = corners.map { $0.x }.max() ?? 0
            let minY = corners.map { $0.y }.min() ?? 0
            let maxY = corners.map { $0.y }.max() ?? 0
            
            let rectX = Double(minX)
            let rectY = Double(minY)
            let rectW = Double(maxX - minX)
            let rectH = Double(maxY - minY)
            
            items.append(OCRBoxItem(text: text, x: rectX, y: rectY, w: rectW, h: rectH))
        }
        
        return OCRResult(
            text: lines.joined(separator: "\n"),
            image_width: W,
            image_height: H,
            boxes: items
        )
    }
    
    private static func imagePixelSize(from data: Data) -> (Int, Int)? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = (props[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
              let h = (props[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue
        else {
            return nil
        }
        return (w, h)
    }
}
