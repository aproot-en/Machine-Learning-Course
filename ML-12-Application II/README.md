# ML-12-Application II 

ML Visual Lab is an interactive web app for learning machine learning and deep learning through 3D simulations. Students can adjust algorithm parameters, build neural networks, train supported models, inspect intermediate calculations, and compare experimental results. The interface supports Thai and English. Computation runs locally in the browser.

## Web Application Preview

### Home Page

The home page introduces the application with an interactive 3D neural network and navigation to the learning labs.

![ML Visual Lab home page](image-3D.png)

### CNN Lab

The CNN lab displays the input image, convolution and pooling feature maps, dense layers, and output probabilities for digits 0–9. Controls allow students to edit the architecture, train the model, draw or upload an image, and inspect calculations.

![CNN lab showing input, feature maps, neural connections, and predictions](image-3D-CNN.png)

Metrics in the screenshot describe that particular experiment and vary with the model, input, and training settings.

## Supported Algorithms and Labs

| Area | Algorithms / models | Main features |
| --- | --- | --- |
| Classification and regression | KNN, SVM, Linear Regression, Logistic Regression | Distances, votes, kernels, margins, residuals, and decision boundaries |
| Clustering | K-Means, DBSCAN | Cluster updates, neighborhoods, core/border/noise points |
| Tree models | Decision Tree, Random Forest | Tree growth, decision paths, bootstrap, and feature importance |
| Optimization | Gradient Descent | Loss surfaces and optimizer trajectory comparison |
| Neural networks | MLP, CNN | Architecture editing, training, activations, weights, and feature maps |
| Sequence models | RNN, LSTM | Unrolled networks, recurrent states, gates, and gradient inspection |
| Attention | Transformer | Multi-head attention, Q/K/V inspection, and attention heatmaps |
| Vision architectures | VGG16, ResNet50, DenseNet121, InceptionV3, MobileNetV2, ViT | Compact executable teaching models and full-architecture references |

The Algorithms gallery contains 20 runnable algorithm/model labs. Additional pages provide a Dataset Explorer, Model Comparison, and 21 lessons with quizzes and selectable learning levels.

### Dataset and Training Setup

The following settings describe the bundled pretrained CNN in `src/data/cnnPretrained.json` and its training script, `scripts/pretrain-cnn.ts`.

| Setting | Recorded value |
| --- | --- |
| Dataset | UCI optical handwritten digits, distributed through scikit-learn's `load_digits` |
| Original images | 1,797 grayscale images, 8 × 8 pixels, values 0–16 |
| Model input / output | 16 × 16 grayscale images / 10 classes, digits 0–9 |
| Architecture | Conv(6, 3×3, same) → MaxPool(2×2) → Conv(8, 3×3, valid) → MaxPool(2×2) → Flatten → Dense(24) → Output(10) |
| Hidden activations | ReLU |
| Train / test | 1,438 / 359 images; stratified 80% / 20% split, seed 42 |
| Optimizer / learning rate | Adam / 0.003 |
| Batch size / epochs | 16 / 8 |
| Model initialization seed | 7 |
| Recorded held-out test accuracy | **97.49%** |
| Execution | TypeScript computation in the browser; no Python backend required |

Other available datasets include Iris, Wine, Breast Cancer, and seeded synthetic datasets. The Dataset Explorer also supports CSV import.

### Model Scope

- The six vision architecture labs use compact 16 × 16 grayscale teaching models with reduced channel widths and frozen, seeded random backbones. Only their classifier heads train. Full reference architectures are documented separately.
- These vision labs do not load ImageNet weights or train complete production architectures. Their metrics describe the teaching experiments.
- The Transformer lab uses seeded random, untrained weights to demonstrate attention.
- The dedicated 3D Playground page is a planned feature. Working labs are available through the Algorithms gallery.

## Project Structure

```text
ML-12-3D-Simulation/
├── index.html                  # Main HTML entry point
├── ML_App.bat                  # Windows launcher
├── package.json                # Dependencies and npm scripts
├── package-lock.json           # Locked dependency versions
├── vite.config.ts              # Development and build configuration
├── tailwind.config.js          # Theme and styling configuration
├── tsconfig.json               # TypeScript configuration
├── src/
│   ├── main.tsx                # React entry point and hash routing
│   ├── App.tsx                 # Application routes
│   ├── components/             # Shared interface components
│   ├── content/                # Algorithm and lesson content
│   ├── data/                   # Datasets, generators, image input, CNN weights
│   ├── engine3d/               # 3D stage, controls, meshes, and labels
│   ├── features/               # Labs, builders, datasets, comparison, lessons
│   ├── i18n/                   # Language and learning-level settings
│   ├── lib/                    # Shared utilities
│   ├── ml/                     # Computation engines and unit tests
│   └── pages/                  # Home, algorithm gallery, and information pages
├── scripts/
│   └── pretrain-cnn.ts          # Train and export the default CNN
├── prototype-standalone.html   # Earlier standalone prototype
├── AUDIT.md                    # Technical verification and known limitations
├── image-3D.png                # Home page screenshot
└── image-3D-CNN.png            # CNN lab screenshot
```

## Tech Stack

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![Three.js](https://img.shields.io/badge/Three.js-000000?style=for-the-badge&logo=threedotjs&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)
![Vitest](https://img.shields.io/badge/Vitest-6E9F18?style=for-the-badge&logo=vitest&logoColor=white)

React Three Fiber, Drei, and React Three Postprocessing provide the 3D rendering layer. Zustand manages application state, React Router handles navigation, Framer Motion provides animation, Plotly.js displays charts, and KaTeX renders equations.

## Installation on Windows

Run the following PowerShell commands from the project root. Install Node.js with npm first. The pinned Vite version supports Node.js `^18.0.0 || ^20.0.0 || >=22.0.0`. The browser must support WebGL.

### 1. Check Node.js and npm

```powershell
node --version
npm --version
```

### 2. Install Dependencies

```powershell
npm ci
```

### 3. Run the App

```powershell
npm run dev
```

Open the URL printed in the terminal, normally [http://localhost:5173](http://localhost:5173). Press `Ctrl+C` to stop the server.

Alternatively, double-click `ML_App.bat`. It checks for Node.js and npm, installs dependencies if Vite is missing, and starts the development server with the browser opening automatically. No API key or separate database is required.

## Development and Verification

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start the development server |
| `npm run typecheck` | Check TypeScript types |
| `npm test` | Run the computation engine tests with Vitest |
| `npm run build` | Check TypeScript and create the production build in `dist/` |
| `npm run preview` | Preview an existing production build locally |

The test suite includes algorithm calculations, numerical gradient checks, training behavior, tensor shapes, attention normalization, and train-only preprocessing. See [AUDIT.md](AUDIT.md) for verification results and remaining limitations, including browser/WebGL checks.

To regenerate the bundled CNN weights with the locked dependencies:

```powershell
npx vite-node scripts/pretrain-cnn.ts
```

This command overwrites `src/data/cnnPretrained.json` with newly trained weights and metrics.

## Production Build

```powershell
npm run build
npm run preview
```

Publish the contents of `dist/` to a static host. The application uses hash-based routes, such as `/#/algorithms/knn`.

For GitHub Pages hosted under a repository path, build with a relative asset base:

```powershell
npm run build -- --base=./
```

Configure the hosting workflow to publish the resulting `dist/` contents. Keep both preview images in the repository root so the README image links display correctly on GitHub


