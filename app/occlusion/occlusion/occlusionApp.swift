import SwiftUI

@main
struct OcclusionApp: App {
    let gazeDetector = GazeDetector()
    lazy var cameraManager = CameraManager(gazeDetector: gazeDetector)
    
    init() {
        cameraManager.start() // todo: probably should end it
    }
    
    var body: some Scene {
        MenuBarExtra("Occlusion", systemImage: "eye.slash.circle.fill") {
            ContentView()
        }
    }
}
