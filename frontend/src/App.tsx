import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { MainLayout } from './layouts/MainLayout';
import { Dashboard } from './pages/Dashboard';
import { DocumentsPage } from './pages/Documents';
import { AssistantPage } from './pages/Assistant';
import { ComparisonPage } from './pages/Comparison';
import { EvaluationPage } from './pages/Evaluation';
import { NotFoundPage } from './pages/NotFound';

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="documents" element={<DocumentsPage />} />
          <Route path="assistant" element={<AssistantPage />} />
          <Route path="comparison" element={<ComparisonPage />} />
          <Route path="evaluation" element={<EvaluationPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
