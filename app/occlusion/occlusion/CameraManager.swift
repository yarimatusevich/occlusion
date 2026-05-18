import AVFoundation

class CameraManager: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    var frameCount = 0
    var device: AVCaptureDevice?
    var captureSesssion: AVCaptureSession
    var gazeDetector: GazeDetector
    
    init(gazeDetector: GazeDetector) {
        self.captureSesssion = AVCaptureSession()
        self.device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
        self.gazeDetector = gazeDetector
    }
    
    func setUpCaptureSessionVideoInput() {
        // if user has no camera or camera failed to init, return
        guard let videoDevice = self.device else {
            return
        }
        
        guard let videoDeviceInput = try? AVCaptureDeviceInput(device: videoDevice) else {
            return
        }
        
        self.captureSesssion.addInput(videoDeviceInput)
    }
    
    func setUpCaptureSessionOutput() {
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera.frame.queue"))
        
        guard self.captureSesssion.canAddOutput(videoOutput) else { return }
        self.captureSesssion.addOutput(videoOutput)
    }
    
    func configCaptureSession() {
        // todo: break out of here if any of those functions fail
        self.captureSesssion.sessionPreset = .low
        self.captureSesssion.beginConfiguration()
        setUpCaptureSessionVideoInput()
        setUpCaptureSessionOutput()
        self.captureSesssion.commitConfiguration()
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        frameCount += 1
        guard frameCount % 10 == 0 else { return }
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        gazeDetector.detectGaze(pixelBuffer)
    }
    
    func start() {
        configCaptureSession()
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSesssion.startRunning()
        }
    }
}
