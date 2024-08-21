import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './pages/Home';
import PlaylistPage from './pages/PlaylistPage';
import Preloader from './components/Preloader';

function App() {
  return (
    <Router>
      <Preloader />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/playlist" element={<PlaylistPage />} />
      </Routes>
    </Router>
  );
}

export default App;
