import { lazy, Suspense } from 'react';
import { Route, Routes, useParams } from 'react-router-dom';
import { Layout } from './components/Layout';
import Home from './pages/Home';
import Algorithms from './pages/Algorithms';
import { About, Planned } from './pages/InfoPages';

const KnnLab = lazy(() => import('./features/knn/KnnLab'));
const NetworkBuilder = lazy(() => import('./features/mlp/NetworkBuilder'));
const SvmLab = lazy(() => import('./features/svm/SvmLab'));
const RegressionLab = lazy(() => import('./features/regression/RegressionLab'));
const ClusterLab = lazy(() => import('./features/cluster/ClusterLab'));
const OptimLab = lazy(() => import('./features/optim/OptimLab'));
const TreeLab = lazy(() => import('./features/tree/TreeLab'));
const CnnLab = lazy(() => import('./features/cnn/CnnLab'));
const SequenceLab = lazy(() => import('./features/sequence/SequenceLab'));
const TransformerLab = lazy(() => import('./features/transformer/TransformerLab'));
const ComparePage = lazy(() => import('./features/compare/ComparePage'));
const DatasetExplorer = lazy(() => import('./features/datasets/DatasetExplorer'));
const LearnPage = lazy(() => import('./features/learn/LearnPage'));
const VisionLab = lazy(() => import('./features/vision/VisionLab'));

function AlgorithmDetail() {
  const { slug } = useParams();
  switch (slug) {
    case 'vgg16':
    case 'resnet50':
    case 'densenet121':
    case 'inceptionv3':
    case 'mobilenetv2':
    case 'vit': return <VisionLab key={slug} modelId={slug} />;
    case 'knn': return <KnnLab />;
    case 'mlp': return <NetworkBuilder />;
    case 'svm': return <SvmLab />;
    case 'linear-regression': return <RegressionLab key="lin" initial="linear" />;
    case 'logistic-regression': return <RegressionLab key="log" initial="logistic" />;
    case 'kmeans': return <ClusterLab key="km" initial="kmeans" />;
    case 'dbscan': return <ClusterLab key="db" initial="dbscan" />;
    case 'gradient-descent': return <OptimLab />;
    case 'decision-tree': return <TreeLab key="dt" initial="tree" />;
    case 'random-forest': return <TreeLab key="rf" initial="forest" />;
    case 'cnn': return <CnnLab />;
    case 'rnn': return <SequenceLab key="rnn" initial="rnn" />;
    case 'lstm': return <SequenceLab key="lstm" initial="lstm" />;
    case 'transformer': return <TransformerLab />;
    default: return <Planned page={slug ?? ''} />;
  }
}

const Loading = () => <div className="grid h-[60vh] place-items-center font-mono text-sm text-muted">loading…</div>;

export default function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="algorithms" element={<Algorithms />} />
          <Route path="algorithms/:slug" element={<AlgorithmDetail />} />
          <Route path="network-builder" element={<NetworkBuilder />} />
          <Route path="model-builder" element={<NetworkBuilder mode="model" />} />
          <Route path="cnn-lab" element={<CnnLab />} />
          <Route path="optimization" element={<OptimLab />} />
          <Route path="playground" element={<Planned page="playground" />} />
          <Route path="datasets" element={<DatasetExplorer />} />
          <Route path="compare" element={<ComparePage />} />
          <Route path="learn" element={<LearnPage />} />
          <Route path="learn/:slug" element={<LearnPage />} />
          <Route path="about" element={<About />} />
          <Route path="*" element={<Planned page="404" />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
