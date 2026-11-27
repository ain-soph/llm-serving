// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "IndexTTSServing",
    platforms: [.macOS(.v26)],
    products: [.executable(name: "indextts-worker", targets: ["IndexTTSServing"])],
    dependencies: [
        .package(url: "https://github.com/xocialize/mlx-indextts2-swift", revision: "1b8298b3d5eb8f2e6835cab08203c97f43e00e61"),
        .package(url: "https://github.com/xocialize/mlx-swift", exact: "0.32.4"),
    ],
    targets: [
        .executableTarget(
            name: "IndexTTSServing",
            dependencies: [
                .product(name: "MLXIndexTTS2", package: "mlx-indextts2-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
