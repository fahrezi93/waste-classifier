import React, { useState, useRef } from 'react';
import axios from 'axios';

// --- KONFIGURASI URL API UNTUK DEPLOYMENT ---
// Vite akan otomatis menggunakan VITE_API_URL saat di-build di Vercel,
// dan akan menggunakan alamat lokal sebagai cadangan saat development.
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
const API_URL = `${API_BASE_URL}/predict`;
const CORRECTION_API_URL = `${API_BASE_URL}/correct-prediction`;


// --- Kumpulan Ikon SVG untuk Halaman (tidak berubah) ---
const TechIcon = ({ children }) => ( <div className="w-16 h-16 bg-purple-200/10 border border-purple-300/20 rounded-2xl flex items-center justify-center mb-4">{children}</div>);
const PythonTensorFlowIcon = () => (<svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M13.5 3.5H10.5L6.5 8.5L7.5 12L10.5 14H13.5L17.5 9L16.5 5.5L13.5 3.5Z" stroke="#a78bfa" strokeWidth="1.5"/><path d="M10.5 20.5H13.5L17.5 15.5L16.5 12L13.5 10H10.5L6.5 15L7.5 18.5L10.5 20.5Z" stroke="#a78bfa" strokeWidth="1.5"/></svg>);
const FlaskIcon = () => (<svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M6 14.5C6 14.5 9 12 12 12C15 12 18 14.5 18 14.5V9L16 4H8L6 9V14.5Z" stroke="#a78bfa" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><path d="M9 4.00002C9.66667 3.00002 11.1 2.00002 12 2.00002C12.9 2.00002 14.3333 3.00002 15 4.00002" stroke="#a78bfa" strokeWidth="1.5" strokeLinecap="round"/></svg>);
const ReactTailwindIcon = () => (<svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="2" stroke="#a78bfa" strokeWidth="1.5"/><ellipse cx="12" cy="12" rx="10" ry="4" stroke="#a78bfa" strokeWidth="1.5"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(60 12 12)" stroke="#a78bfa" strokeWidth="1.5"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(120 12 12)" stroke="#a78bfa" strokeWidth="1.5"/></svg>);
const CloudIcon = () => (<svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 13.8214 2.48697 15.5291 3.33782 17" stroke="#a78bfa" strokeWidth="1.5" strokeLinecap="round"/><path d="M19.0002 17C20.1048 17 21.0002 16.1046 21.0002 15C21.0002 13.8954 20.1048 13 19.0002 13C18.7849 13 18.5783 13.027 18.3825 13.0784C18.156 10.7231 16.3153 8.87754 14.0002 8.5C11.5002 8.5 9.50016 10.5 9.50016 13C9.50016 13.0973 9.5049 13.1933 9.51412 13.2878C8.66447 13.6369 8.00016 14.4754 8.00016 15.5C8.00016 16.8807 9.11944 18 10.5002 18H17.0002" stroke="#a78bfa" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>);
const UploadIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-gray-400 group-hover:text-purple-500 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" /></svg>);
const LeafIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 inline-block mr-2" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" /></svg>);
const RecycleIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 inline-block mr-2" viewBox="0 0 20 20" fill="currentColor"><path d="M10 2a8 8 0 100 16 8 8 0 000-16zm0 14a6 6 0 110-12 6 6 0 010 12z" /><path d="M10 4a1 1 0 00-1 1v4a1 1 0 102 0V5a1 1 0 00-1-1z" /><path d="M10 12a1 1 0 00-1 1v1a1 1 0 102 0v-1a1 1 0 00-1-1z" /></svg>);
const AksaraIcon = () => (<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M15 9L9 15" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M9 9L15 15" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>);

// --- Komponen Halaman Classifier ---
const ClassifierPage = ({ onBack }) => {
  // ... (semua state dan fungsi di dalam komponen ini tidak berubah)
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);
  const [showCorrection, setShowCorrection] = useState(false);
  const [correctionSent, setCorrectionSent] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && (selectedFile.type === "image/jpeg" || selectedFile.type === "image/png")) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError('');
      setShowCorrection(false);
      setCorrectionSent(false);
    } else {
      setError('Silakan pilih file gambar (JPG atau PNG).');
      setFile(null);
      setPreview(null);
    }
  };

  const handleSubmit = async () => {
    if (!file) return;
    setIsLoading(true);
    setError('');
    setShowCorrection(false);
    setCorrectionSent(false);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await axios.post(API_URL, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      setResult(response.data);
    } catch (err) {
      setError('Gagal melakukan klasifikasi. Pastikan server backend berjalan.');
      console.error("Classification Error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCorrection = async (correctLabel) => {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    formData.append('correct_label', correctLabel);
    try {
        await axios.post(CORRECTION_API_URL, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
        setCorrectionSent(true);
        setShowCorrection(false);
    } catch (err) {
        console.error("Correction Error:", err);
        setError("Gagal mengirim koreksi ke server.");
    }
  };

  const getResultStyles = (prediction) => ({
    barColor: prediction === 'Organik' ? 'bg-green-500' : 'bg-sky-500',
    bgColor: prediction === 'Organik' ? 'bg-green-500/10' : 'bg-sky-500/10',
    textColor: prediction === 'Organik' ? 'text-green-300' : 'text-sky-300',
    borderColor: prediction === 'Organik' ? 'border-green-500/30' : 'border-sky-500/30',
    Icon: prediction === 'Organik' ? LeafIcon : RecycleIcon,
  });

  const styles = result ? getResultStyles(result.prediction) : {};

  return (
    <div className="w-full min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4 animate-fade-in">
       <button onClick={onBack} className="absolute top-5 left-5 text-sm text-gray-400 hover:text-white transition-colors">&larr; Kembali</button>
      <div className="w-full max-w-md bg-gray-800/50 backdrop-blur-sm rounded-2xl shadow-2xl shadow-black/30 p-8 space-y-6 border border-gray-700">
        <header className="text-center"><h1 className="text-3xl font-bold text-gray-100">Waste Classifier</h1><p className="text-gray-400 mt-1">Upload gambar untuk klasifikasi</p></header>
        <div className="w-full h-60 border-2 border-dashed border-gray-600 rounded-lg flex items-center justify-center cursor-pointer hover:border-purple-500 hover:bg-gray-800/60 transition-all group" onClick={() => fileInputRef.current.click()}>
          {preview ? <img src={preview} alt="Preview" className="max-h-full max-w-full object-contain rounded-md" /> : <div className="text-center"><UploadIcon /><p className="mt-2 text-sm text-gray-500">Pilih Gambar</p></div>}
        </div>
        <input type="file" accept="image/jpeg, image/png" onChange={handleFileChange} ref={fileInputRef} className="hidden" />
        <button onClick={handleSubmit} disabled={!file || isLoading} className="w-full bg-purple-600 text-white font-semibold py-3 rounded-lg hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 focus:ring-purple-500">{isLoading ? 'Menganalisis...' : 'Klasifikasikan'}</button>
        {error && <p className="text-red-400 text-sm text-center bg-red-500/10 p-3 rounded-lg">{error}</p>}
        {result && !isLoading && (
          <div className={`pt-4 border-t ${styles.borderColor} animate-fade-in`}>
            <div className={`p-4 rounded-lg text-white text-center ${styles.bgColor}`}><p className="text-2xl font-bold tracking-wider flex items-center justify-center"><styles.Icon />{result.prediction}</p></div>
            <div className="mt-3">
              <div className="w-full bg-gray-700 rounded-full h-2.5 overflow-hidden"><div className={`h-full rounded-full ${styles.barColor}`} style={{ width: `${result.confidence * 100}%` }}></div></div>
              <p className={`text-center text-xs mt-1.5 ${styles.textColor}`}>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
            </div>
            <div className="mt-4 text-center">
              {correctionSent ? (<p className="text-green-400 text-sm">✓ Terima kasih atas masukan Anda!</p>) : showCorrection ? (<div className="space-y-2"><p className="text-sm text-gray-400">Apa klasifikasi yang benar?</p><div className="flex gap-2"><button onClick={() => handleCorrection('Organik')} className="w-full text-sm bg-green-500/20 text-green-300 font-semibold py-2 rounded-lg hover:bg-green-500/40 transition">Organik</button><button onClick={() => handleCorrection('Anorganik')} className="w-full text-sm bg-sky-500/20 text-sky-300 font-semibold py-2 rounded-lg hover:bg-sky-500/40 transition">Anorganik</button></div></div>) : (<button onClick={() => setShowCorrection(true)} className="text-xs text-gray-500 hover:text-white underline">Hasil ini salah?</button>)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// --- Komponen Halaman Landing (tidak berubah) ---
const LandingPage = ({ onStart }) => {
  // ... (JSX untuk LandingPage tetap sama)
  const technologies = [
    { name: "Python + TensorFlow", description: "Kerangka kerja utama untuk training model dan implementasi deep learning.", icon: <PythonTensorFlowIcon /> },
    { name: "Flask API", description: "Backend ringan untuk melayani permintaan dari model Machine Learning.", icon: <FlaskIcon /> },
    { name: "React + Tailwind CSS", description: "Frontend modern dan responsif untuk interaksi pengguna yang dinamis.", icon: <ReactTailwindIcon /> },
    { name: "Cloud Deployment", description: "Hosting yang skalabel untuk penggunaan produksi (Vercel & Render).", icon: <CloudIcon /> },
  ];
  return (
    <main className="bg-gray-900 text-white">
      <section className="w-full min-h-screen flex flex-col items-center justify-center p-8 text-center animated-gradient"><div className="text-focus-in"><h1 className="text-5xl md:text-7xl font-extrabold tracking-tight">Trash Classification</h1><h2 className="text-5xl md:text-7xl font-extrabold tracking-tight text-purple-300">with Deep Learning</h2></div><p className="max-w-2xl mt-6 text-lg text-purple-100/80 text-focus-in" style={{animationDelay: '0.3s'}}>Sebuah cara pintar untuk mengidentifikasi jenis sampah menggunakan Kecerdasan Buatan. Didukung oleh arsitektur MobileNetV2 untuk pengenalan yang akurat dan cerdas.</p><button onClick={onStart} className="mt-10 px-8 py-4 bg-white text-purple-700 font-bold rounded-full shadow-2xl hover:bg-purple-100 transform hover:scale-105 transition-all duration-300 scale-in-center">Coba Classifier ✨</button></section>
      <section className="py-20 px-4 bg-gray-900"><div className="max-w-4xl mx-auto text-center"><h2 className="text-4xl font-bold mb-4">Tentang Proyek Ini</h2><div className="w-24 h-1 bg-purple-500 mx-auto mb-10"></div><div className="bg-gray-800/50 border border-gray-700 rounded-2xl p-8 text-lg text-gray-300 leading-relaxed shadow-lg">Proyek ini menerapkan <span className="font-semibold text-purple-300">Convolutional Neural Networks (CNN)</span> untuk mengklasifikasikan jenis sampah menjadi kategori Organik dan Anorganik dari gambar, menggunakan arsitektur <span className="font-semibold text-purple-300">MobileNetV2</span> yang kuat. Tujuannya adalah untuk menunjukkan bagaimana deep learning dapat digunakan dalam aplikasi dunia nyata seperti pengelolaan limbah, edukasi lingkungan, atau sistem daur ulang otomatis. Sistem ini dilatih untuk mengenali berbagai jenis sampah dengan akurasi tinggi melalui antarmuka web yang intuitif.</div></div></section>
      <section className="py-20 px-4"><div className="max-w-6xl mx-auto text-center"><h2 className="text-4xl font-bold mb-4">Teknologi yang Digunakan</h2><div className="w-24 h-1 bg-purple-500 mx-auto mb-10"></div><p className="text-lg text-gray-400 max-w-3xl mx-auto mb-16">Dibangun dengan teknologi deep learning dan web modern untuk memberikan klasifikasi gambar yang cepat dan akurat.</p><div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-10">{technologies.map((tech) => (<div key={tech.name} className="flex flex-col items-center p-6 bg-gray-800/50 border border-gray-700 rounded-2xl shadow-lg hover:bg-gray-800 transition-colors"><TechIcon>{tech.icon}</TechIcon><h3 className="text-xl font-bold mb-2">{tech.name}</h3><p className="text-gray-400 text-sm">{tech.description}</p></div>))}</div></div></section>
      <footer className="text-center py-8 text-gray-500 border-t border-gray-800">Dibuat dengan 💜 oleh Anda.</footer>
    </main>
  );
};

// --- Komponen Aplikasi Utama (tidak berubah) ---
export default function App() {
  const [showClassifier, setShowClassifier] = useState(false);
  return (
    <div>
      {showClassifier ? (<ClassifierPage onBack={() => setShowClassifier(false)} />) : (<LandingPage onStart={() => setShowClassifier(true)} />)}
      <a href="https://aksara.ai/" target="_blank" rel="noopener noreferrer" className="fixed bottom-6 right-6 bg-purple-600 text-white p-4 rounded-full shadow-lg hover:bg-purple-700 transform hover:scale-110 transition-all duration-300 z-50 flex items-center group hover:pr-6" title="Kunjungi Aksara AI"><AksaraIcon /><span className="max-w-0 overflow-hidden group-hover:max-w-48 transition-all duration-500 ease-in-out ml-0 group-hover:ml-3 whitespace-nowrap text-sm font-medium">Coba tanyakan ke Aksara AI!</span></a>
    </div>
  );
}
