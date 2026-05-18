import CoreML
import Vision
import CoreImage

class GazeDetector {
    var model: OcclusionModel?
    var faceDectectionRequest = VNDetectFaceRectanglesRequest()
    var context = CIContext() // CIImage context used to convert CIImage to CVBuffer
    
    init() {
        self.model = try? OcclusionModel()
    }
    
    func convertCIImageToCVBuffer(_ image: CIImage) -> CVPixelBuffer? {
        let width = Int(image.extent.width)
        let height = Int(image.extent.height)
        
        var pixelBuffer: CVPixelBuffer?
        
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        self.context.render(image, to: buffer)
        
        return buffer
    }
    
    func detectFaceAndCrop(_ inputFrameBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        let handler = VNImageRequestHandler(cvPixelBuffer: inputFrameBuffer)
        try? handler.perform([self.faceDectectionRequest])
        
        guard let results = self.faceDectectionRequest.results, let face = results.first else {
            // no face detected
            return nil
        }
        
        // getting bounding box coordinates and width, height
        let inputFrameWidth = CVPixelBufferGetWidth(inputFrameBuffer)
        let inputFrameHeight = CVPixelBufferGetHeight(inputFrameBuffer)
        let boundBox = face.boundingBox
        
        // getting bounding box pixel cords and converting input frame buffer to CIImage for cropping
        let faceRect = VNImageRectForNormalizedRect(boundBox, inputFrameWidth, inputFrameHeight)
        let inputFrameImage = CIImage(cvPixelBuffer: inputFrameBuffer)
        
        // cropping image to bound box
        let croppedImage = inputFrameImage.cropped(to: faceRect)
        
        let translatedImage = croppedImage.transformed(by: CGAffineTransform(translationX: -croppedImage.extent.origin.x,
                                                                              y: -croppedImage.extent.origin.y))
        
        let scaleX = 128 / croppedImage.extent.width
        let scaleY = 128 / croppedImage.extent.height
        
        let resizedCroppedImage = croppedImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        return convertCIImageToCVBuffer(resizedCroppedImage)
    }
    
    func detectGaze(_ inputFrameBuffer: CVPixelBuffer) {
        guard let croppedFaceBuffer = detectFaceAndCrop(inputFrameBuffer) else {
            print("No face detected")
            return
        }
        
        guard let p = try? self.model?.prediction(x: croppedFaceBuffer) else {
            print("Model prediction failed")
            return
        }
        
        print(p)
    }
}
