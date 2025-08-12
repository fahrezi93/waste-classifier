import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';

// URL backend Flask Anda.
const API_URL = 'http://127.0.0.1:5000/predict';
const CORRECTION_API_URL = 'http://127.0.0.1:5000/correct-prediction';

// --- DATA UNTUK HALAMAN ---
const wasteData = {
  organik: { title: "Sampah Organik", description: "Sampah yang berasal dari sisa makhluk hidup dan mudah terurai secara alami oleh mikroorganisme.", handling: "Olah menjadi pupuk kompos, gunakan sebagai pakan ternak, atau manfaatkan untuk budidaya maggot.", examples: ["Sisa makanan & sayuran", "Daun kering & ranting", "Kotoran hewan"], icon: "🌿", color: "green" },
  anorganik: { title: "Sampah Anorganik", description: "Sampah yang sulit atau bahkan tidak bisa terurai, namun sebagian besar dapat didaur ulang.", handling: "Pisahkan berdasarkan jenisnya (plastik, kertas, dll.), setorkan ke bank sampah terdekat, atau gunakan kembali (reuse).", examples: ["Botol plastik & kresek", "Kaleng minuman & kaca", "Kertas & kardus"], icon: "♻️", color: "sky" },
};

const wasteTypesData = {
  plastik: { name: "Plastik", icon: "♻️", items: [ { name: "Gelas Plastik", img: "https://placehold.co/400x400/EBF4FF/7C8B9E?text=Gelas+Plastik" }, { name: "Botol Biru Muda", img: "https://placehold.co/400x400/D6EFFF/7C8B9E?text=Botol+Biru" }, { name: "Botol Warna", img: "https://placehold.co/400x400/FFEBF2/7C8B9E?text=Botol+Warna" }, { name: "Botol Bening", img: "https://placehold.co/400x400/F0F9FF/7C8B9E?text=Botol+Bening" } ]},
  kertas: { name: "Kertas", icon: "📄", items: [ { name: "Koran", img: "https://placehold.co/400x400/F3F4F6/7C8B9E?text=Koran" }, { name: "Kardus", img: "https://placehold.co/400x400/EFEBE9/7C8B9E?text=Kardus" }, { name: "Majalah", img: "https://placehold.co/400x400/EFF6FF/7C8B9E?text=Majalah" }, { name: "Buku", img: "https://placehold.co/400x400/F9FAFB/7C8B9E?text=Buku" } ]},
  logam: { name: "Besi & Logam", icon: "🥫", items: [ { name: "Kaleng Aluminium", img: "https://placehold.co/400x400/E5E7EB/7C8B9E?text=Kaleng" }, { name: "Tutup Botol", img: "https://placehold.co/400x400/D1D5DB/7C8B9E?text=Tutup+Botol" } ]},
  kaca: { name: "Botol Kaca", icon: "🍾", items: [ { name: "Botol Bening", img: "https://placehold.co/400x400/F9FAFB/7C8B9E?text=Botol+Kaca" }, { name: "Botol Coklat", img: "https://placehold.co/400x400/EFEBE9/7C8B9E?text=Botol+Coklat" } ]}
};

// --- Kumpulan Ikon SVG ---
const TechIcon = ({ children }) => ( <div className="w-16 h-16 bg-purple-200/10 border border-purple-300/20 rounded-2xl flex items-center justify-center mb-4">{children}</div>);
const PythonTensorFlowIcon = () => (<svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M13.5 3.5H10.5L6.5 8.5L7.5 12L10.5 14H13.5L17.5 9L16.5 5.5L13.5 3.5Z" stroke="#a78bfa" strokeWidth="1.5"/><path d="M10.5 20.5H13.5L17.5 15.5L16.5 12L13.5 10H10.5L6.5 15L7.5 18.5L10.5 20.5Z" stroke="#a78bfa" strokeWidth="1.5"/></svg>);
const FlaskIcon = () => (<svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M6 14.5C6 14.5 9 12 12 12C15 12 18 14.5 18 14.5V9L16 4H8L6 9V14.5Z" stroke="#a78bfa" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><path d="M9 4.00002C9.66667 3.00002 11.1 2.00002 12 2.00002C12.9 2.00002 14.3333 3.00002 15 4.00002" stroke="#a78bfa" strokeWidth="1.5" strokeLinecap="round"/></svg>);
const ReactTailwindIcon = () => (<svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="2" stroke="#a78bfa" strokeWidth="1.5"/><ellipse cx="12" cy="12" rx="10" ry="4" stroke="#a78bfa" strokeWidth="1.5"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(60 12 12)" stroke="#a78bfa" strokeWidth="1.5"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(120 12 12)" stroke="#a78bfa" strokeWidth="1.5"/></svg>);
const CloudIcon = () => (<svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 13.8214 2.48697 15.5291 3.33782 17" stroke="#a78bfa" strokeWidth="1.5" strokeLinecap="round"/><path d="M19.0002 17C20.1048 17 21.0002 16.1046 21.0002 15C21.0002 13.8954 20.1048 13 19.0002 13C18.7849 13 18.5783 13.027 18.3825 13.0784C18.156 10.7231 16.3153 8.87754 14.0002 8.5C11.5002 8.5 9.50016 10.5 9.50016 13C9.50016 13.0973 9.5049 13.1933 9.51412 13.2878C8.66447 13.6369 8.00016 14.4754 8.00016 15.5C8.00016 16.8807 9.11944 18 10.5002 18H17.0002" stroke="#a78bfa" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>);
const UploadIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-gray-400 group-hover:text-purple-500 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" /></svg>);
const CameraIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" /><path strokeLinecap="round" strokeLinejoin="round" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" /></svg>);
const LeafIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 inline-block mr-2" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" /></svg>);
const RecycleIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 inline-block mr-2" viewBox="0 0 20 20" fill="currentColor"><path d="M10 2a8 8 0 100 16 8 8 0 000-16zm0 14a6 6 0 110-12 6 6 0 010 12z" /><path d="M10 4a1 1 0 00-1 1v4a1 1 0 102 0V5a1 1 0 00-1-1z" /><path d="M10 12a1 1 0 00-1 1v1a1 1 0 102 0v-1a1 1 0 00-1-1z" /></svg>);
const AksaraIcon = () => (<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M15 9L9 15" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M9 9L15 15" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>);

// --- Komponen Kamera ---
const CameraView = ({ onCapture, onCancel }) => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    useEffect(() => {
        let stream = null;
        const startCamera = async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
                if (videoRef.current) { videoRef.current.srcObject = stream; }
            } catch (err) { console.error("Error accessing camera:", err); alert("Tidak bisa mengakses kamera. Pastikan Anda memberikan izin."); onCancel(); }
        };
        startCamera();
        return () => { stream?.getTracks().forEach(track => track.stop()); };
    }, [onCancel]);
    const handleCapture = () => {
        const video = videoRef.current; const canvas = canvasRef.current;
        if (video && canvas) {
            const context = canvas.getContext('2d');
            canvas.width = video.videoWidth; canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
            canvas.toBlob(blob => { const file = new File([blob], "capture.jpg", { type: "image/jpeg" }); onCapture(file); }, 'image/jpeg');
        }
    };
    return (<div className="w-full relative"><video ref={videoRef} autoPlay playsInline className="w-full h-60 rounded-lg object-cover bg-gray-900"></video><canvas ref={canvasRef} className="hidden"></canvas><div className="absolute bottom-4 left-0 right-0 flex justify-center gap-4"><button onClick={handleCapture} className="p-4 bg-purple-600 rounded-full text-white shadow-lg focus:outline-none focus:ring-2 focus:ring-purple-400"><CameraIcon /></button><button onClick={onCancel} className="px-4 py-2 bg-gray-700 text-white text-sm rounded-full shadow-lg">Batal</button></div></div>);
};

// --- Komponen Halaman Classifier ---
const ClassifierPage = ({ onBack }) => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);
  const [showCorrection, setShowCorrection] = useState(false);
  const [correctionSent, setCorrectionSent] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const resetState = () => { setResult(null); setError(''); setShowCorrection(false); setCorrectionSent(false); };
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && (selectedFile.type === "image/jpeg" || selectedFile.type === "image/png")) { setFile(selectedFile); setPreview(URL.createObjectURL(selectedFile)); resetState(); } 
    else { setError('Silakan pilih file gambar (JPG atau PNG).'); setFile(null); setPreview(null); }
  };
  const handleCapture = useCallback((capturedFile) => { setFile(capturedFile); setPreview(URL.createObjectURL(capturedFile)); setShowCamera(false); resetState(); }, []);
  const handleSubmit = async () => {
    if (!file) return; setIsLoading(true); resetState(); const formData = new FormData(); formData.append('file', file);
    try { const response = await axios.post(API_URL, formData, { headers: { 'Content-Type': 'multipart/form-data' } }); setResult(response.data); } 
    catch (err) { setError('Gagal melakukan klasifikasi. Pastikan server backend berjalan.'); console.error("Classification Error:", err); } 
    finally { setIsLoading(false); }
  };
  const handleCorrection = async (correctLabel) => {
    if (!file) return; const formData = new FormData(); formData.append('file', file); formData.append('correct_label', correctLabel);
    try { await axios.post(CORRECTION_API_URL, formData, { headers: { 'Content-Type': 'multipart/form-data' } }); setCorrectionSent(true); setShowCorrection(false); } 
    catch (err) { console.error("Correction Error:", err); setError("Gagal mengirim koreksi ke server."); }
  };
  const getResultStyles = (prediction) => ({ barColor: prediction === 'Organik' ? 'bg-green-500' : 'bg-sky-500', bgColor: prediction === 'Organik' ? 'bg-green-500/10' : 'bg-sky-500/10', textColor: prediction === 'Organik' ? 'text-green-300' : 'text-sky-300', borderColor: prediction === 'Organik' ? 'border-green-500/30' : 'border-sky-500/30', Icon: prediction === 'Organik' ? LeafIcon : RecycleIcon, });
  const styles = result ? getResultStyles(result.prediction) : {};
  return (<div className="w-full min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4 animate-fade-in"><button onClick={onBack} className="absolute top-5 left-5 text-sm text-gray-400 hover:text-white transition-colors">&larr; Kembali</button><div className="w-full max-w-md bg-gray-800/50 backdrop-blur-sm rounded-2xl shadow-2xl shadow-black/30 p-8 space-y-6 border border-gray-700"><header className="text-center"><h1 className="text-3xl font-bold text-gray-100">Waste Classifier</h1><p className="text-gray-400 mt-1">Upload gambar untuk klasifikasi</p></header>{showCamera ? (<CameraView onCapture={handleCapture} onCancel={() => setShowCamera(false)} />) : (<div className="w-full h-60 border-2 border-dashed border-gray-600 rounded-lg flex items-center justify-center cursor-pointer hover:border-purple-500 hover:bg-gray-800/60 transition-all group" onClick={() => fileInputRef.current.click()}>{preview ? <img src={preview} alt="Preview" className="max-h-full max-w-full object-contain rounded-md" /> : <div className="text-center"><UploadIcon /><p className="mt-2 text-sm text-gray-500">Pilih Gambar</p></div>}</div>)}<input type="file" accept="image/jpeg, image/png" onChange={handleFileChange} ref={fileInputRef} className="hidden" /><div className="flex gap-4"><button onClick={() => setShowCamera(true)} className="w-full flex items-center justify-center bg-gray-700 text-white font-semibold py-3 rounded-lg hover:bg-gray-600 transition duration-300"><CameraIcon /> Kamera</button><button onClick={handleSubmit} disabled={!file || isLoading} className="w-full bg-purple-600 text-white font-semibold py-3 rounded-lg hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition duration-300">{isLoading ? 'Menganalisis...' : 'Klasifikasikan'}</button></div>{error && <p className="text-red-400 text-sm text-center bg-red-500/10 p-3 rounded-lg">{error}</p>}{result && !isLoading && (<div className={`pt-4 border-t ${styles.borderColor} animate-fade-in`}><div className={`p-4 rounded-lg text-white text-center ${styles.bgColor}`}><p className="text-2xl font-bold tracking-wider flex items-center justify-center"><styles.Icon />{result.prediction}</p></div><div className="mt-3"><div className="w-full bg-gray-700 rounded-full h-2.5 overflow-hidden"><div className={`h-full rounded-full ${styles.barColor}`} style={{ width: `${result.confidence * 100}%` }}></div></div><p className={`text-center text-xs mt-1.5 ${styles.textColor}`}>Confidence: {(result.confidence * 100).toFixed(2)}%</p></div><div className="mt-4 text-center">{correctionSent ? (<p className="text-green-400 text-sm">✓ Terima kasih atas masukan Anda!</p>) : showCorrection ? (<div className="space-y-2"><p className="text-sm text-gray-400">Apa klasifikasi yang benar?</p><div className="flex gap-2"><button onClick={() => handleCorrection('Organik')} className="w-full text-sm bg-green-500/20 text-green-300 font-semibold py-2 rounded-lg hover:bg-green-500/40 transition">Organik</button><button onClick={() => handleCorrection('Anorganik')} className="w-full text-sm bg-sky-500/20 text-sky-300 font-semibold py-2 rounded-lg hover:bg-sky-500/40 transition">Anorganik</button></div></div>) : (<button onClick={() => setShowCorrection(true)} className="text-xs text-gray-500 hover:text-white underline">Hasil ini salah?</button>)}</div></div>)}</div></div>);
};

// --- Komponen Halaman Jenis Sampah ---
const WasteTypesPage = ({ onSelectCategory, onBack, onGoToClassifier }) => {
    return (<div className="w-full min-h-screen bg-gray-900 text-white flex flex-col items-center p-8 animate-fade-in"><header className="w-full max-w-4xl mb-12"><button onClick={onBack} className="text-sm text-gray-400 hover:text-white transition-colors mb-4">&larr; Kembali ke Halaman Utama</button><h1 className="text-4xl font-bold text-gray-100">Jenis Sampah</h1><p className="text-gray-400 mt-2">Pilih kategori untuk melihat contoh, atau langsung coba classifier.</p></header><div className="w-full max-w-4xl grid grid-cols-2 md:grid-cols-4 gap-6">{Object.keys(wasteTypesData).map(key => (<button key={key} onClick={() => onSelectCategory(key)} className="bg-gray-800/50 border border-gray-700 rounded-2xl p-6 text-center hover:bg-gray-800 hover:border-purple-500 transition-all duration-300"><div className="text-5xl mb-4">{wasteTypesData[key].icon}</div><h3 className="font-semibold text-lg">{wasteTypesData[key].name}</h3></button>))}</div><div className="mt-12 w-full max-w-4xl text-center"><button onClick={onGoToClassifier} className="px-8 py-4 bg-purple-600 text-white font-bold rounded-full shadow-lg hover:bg-purple-700 transform hover:scale-105 transition-all duration-300">Coba Classifier Sekarang &rarr;</button></div></div>);
};

// --- Komponen Halaman Detail Jenis Sampah ---
const WasteDetailPage = ({ category, onBack }) => {
    const data = wasteTypesData[category];
    return (<div className="w-full min-h-screen bg-gray-900 text-white flex flex-col items-center p-8 animate-fade-in"><header className="w-full max-w-5xl mb-12"><button onClick={onBack} className="text-sm text-gray-400 hover:text-white transition-colors mb-4">&larr; Kembali ke Jenis Sampah</button><h1 className="text-4xl font-bold text-gray-100 flex items-center"><span className="text-5xl mr-4">{data.icon}</span>{data.name}</h1></header><div className="w-full max-w-5xl grid grid-cols-2 md:grid-cols-4 gap-6">{data.items.map(item => (<div key={item.name} className="bg-gray-800/50 border border-gray-700 rounded-2xl overflow-hidden"><img src={item.img} alt={item.name} className="w-full h-48 object-cover"/><p className="p-4 font-semibold text-center">{item.name}</p></div>))}</div></div>);
};

// --- Komponen FAQ Item ---
const FaqItem = ({ question, answer, isOpen, onClick }) => {
  return (<div className="border-b border-purple-300/20 py-4"><button onClick={onClick} className="w-full flex justify-between items-center text-left text-lg font-semibold text-white"><span>{question}</span><svg className={`w-5 h-5 transition-transform duration-300 ${isOpen ? 'transform rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg></button><div className={`overflow-hidden transition-all duration-300 ease-in-out ${isOpen ? 'max-h-screen mt-4' : 'max-h-0'}`}><p className="text-gray-400">{answer}</p></div></div>);
};

// --- Komponen Halaman Landing ---
const LandingPage = ({ onStart }) => {
  const [email, setEmail] = useState('');
  const [openFaq, setOpenFaq] = useState(null);
  const handleSubscribe = (e) => { e.preventDefault(); alert(`Terima kasih! Email ${email} telah didaftarkan.`); setEmail(''); };
  const faqData = [ { q: "Apa tujuan utama dari proyek ini?", a: "Tujuan utamanya adalah untuk menyediakan alat berbasis AI yang mudah digunakan untuk mengklasifikasikan sampah menjadi kategori organik dan anorganik, serta mengedukasi masyarakat tentang pentingnya pengelolaan sampah yang benar." }, { q: "Teknologi apa yang digunakan untuk model klasifikasi?", a: "Model ini dibangun menggunakan arsitektur Convolutional Neural Network (CNN) canggih seperti EfficientNetB4 atau ResNet50V2, dilatih dengan library TensorFlow dan Keras." }, { q: "Seberapa akurat model ini?", a: "Akurasi model terus ditingkatkan. Versi saat ini dilatih pada ribuan gambar dan mencapai akurasi validasi di atas 95%, namun hasilnya bisa bervariasi tergantung pada kualitas dan sudut gambar." }, { q: "Bagaimana cara saya berkontribusi?", a: "Anda bisa berkontribusi dengan menggunakan fitur 'Hasil ini salah?' di halaman klasifikasi. Setiap koreksi yang Anda berikan akan membantu kami mengumpulkan data untuk melatih ulang model agar menjadi lebih pintar di masa depan." } ];
  const technologies = [ { name: "Python + TensorFlow", description: "Kerangka kerja utama untuk training model dan implementasi deep learning.", icon: <PythonTensorFlowIcon /> }, { name: "Flask API", description: "Backend ringan untuk melayani permintaan dari model Machine Learning.", icon: <FlaskIcon /> }, { name: "React + Tailwind CSS", description: "Frontend modern dan responsif untuk interaksi pengguna yang dinamis.", icon: <ReactTailwindIcon /> }, { name: "Cloud Deployment", description: "Hosting yang skalabel untuk penggunaan produksi (Vercel & Render).", icon: <CloudIcon /> }, ];
  return (<main className="bg-gray-900 text-white"><section className="relative w-full min-h-screen flex flex-col items-center justify-center p-8 text-center animated-gradient overflow-hidden"><div className="absolute top-0 left-0 w-full h-full z-0"><div className="absolute -top-20 -left-20 w-72 h-72 bg-white/10 rounded-full filter blur-3xl animate-blob"></div><div className="absolute -bottom-20 -right-20 w-72 h-72 bg-purple-400/10 rounded-full filter blur-3xl animate-blob animation-delay-2000"></div><div className="absolute top-1/2 left-1/4 w-60 h-60 bg-sky-400/10 rounded-full filter blur-3xl animate-blob animation-delay-4000"></div></div><div className="relative z-10"><div className="text-focus-in"><h1 className="text-5xl md:text-7xl font-extrabold tracking-tight">Trash Classification</h1><h2 className="text-5xl md:text-7xl font-extrabold tracking-tight text-purple-300">with Deep Learning</h2></div><p className="max-w-2xl mt-6 text-lg text-purple-100/80 text-focus-in" style={{animationDelay: '0.3s'}}>Sebuah cara pintar untuk mengidentifikasi jenis sampah menggunakan Kecerdasan Buatan. Didukung oleh arsitektur canggih untuk pengenalan yang akurat dan cerdas.</p><button onClick={onStart} className="mt-10 px-8 py-4 bg-white text-purple-700 font-bold rounded-full shadow-2xl hover:bg-purple-100 transform hover:scale-105 transition-all duration-300 scale-in-center">Jelajahi Fitur Kami ✨</button></div></section><section className="py-20 px-4 bg-gray-900"><div className="max-w-4xl mx-auto text-center"><h2 className="text-4xl font-bold mb-4">Tentang Proyek Ini</h2><div className="w-24 h-1 bg-purple-500 mx-auto mb-10"></div><div className="bg-gray-800/50 border border-gray-700 rounded-2xl p-8 text-lg text-gray-300 leading-relaxed shadow-lg">Proyek ini menerapkan <span className="font-semibold text-purple-300">Convolutional Neural Networks (CNN)</span> untuk mengklasifikasikan jenis sampah menjadi kategori Organik dan Anorganik dari gambar, menggunakan arsitektur canggih seperti <span className="font-semibold text-purple-300">EfficientNetB4</span> atau <span className="font-semibold text-purple-300">ResNet50V2</span>. Tujuannya adalah untuk menunjukkan bagaimana deep learning dapat digunakan dalam aplikasi dunia nyata seperti pengelolaan limbah, edukasi lingkungan, atau sistem daur ulang otomatis. Sistem ini dilatih untuk mengenali berbagai jenis sampah dengan akurasi tinggi melalui antarmuka web yang intuitif.</div></div></section><section className="py-20 px-4"><div className="max-w-5xl mx-auto text-center"><h2 className="text-4xl font-bold mb-4">Informasi & Cara Penanganan Sampah</h2><div className="w-24 h-1 bg-purple-500 mx-auto mb-10"></div><div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-left">{Object.values(wasteData).map(data => (<div key={data.title} className={`bg-${data.color}-500/10 border border-${data.color}-500/30 rounded-2xl p-8`}><h3 className={`text-2xl font-bold text-${data.color}-300 mb-4 flex items-center`}><span className="text-3xl mr-3">{data.icon}</span>{data.title}</h3><p className="text-gray-400 mb-6">{data.description}</p><h4 className="font-semibold text-white mb-2">Contoh:</h4><ul className="list-disc list-inside text-gray-400 space-y-1 mb-6">{data.examples.map(example => <li key={example}>{example}</li>)}</ul><h4 className="font-semibold text-white mb-2">Cara Penanganan:</h4><p className="text-gray-400">{data.handling}</p></div>))}</div></div></section><section className="py-20 px-4"><div className="max-w-6xl mx-auto text-center"><h2 className="text-4xl font-bold mb-4">Teknologi yang Digunakan</h2><div className="w-24 h-1 bg-purple-500 mx-auto mb-10"></div><p className="text-lg text-gray-400 max-w-3xl mx-auto mb-16">Dibangun dengan teknologi deep learning dan web modern untuk memberikan klasifikasi gambar yang cepat dan akurat.</p><div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-10">{technologies.map((tech) => (<div key={tech.name} className="flex flex-col items-center p-6 bg-gray-800/50 border border-gray-700 rounded-2xl shadow-lg hover:bg-gray-800 transition-colors"><TechIcon>{tech.icon}</TechIcon><h3 className="text-xl font-bold mb-2">{tech.name}</h3><p className="text-gray-400 text-sm">{tech.description}</p></div>))}</div></div></section><section className="py-20 px-4 bg-gray-900"><div className="max-w-4xl mx-auto text-center"><h2 className="text-4xl font-bold mb-4">Frequently Asked Questions (FAQ)</h2><div className="w-24 h-1 bg-purple-500 mx-auto mb-10"></div><div className="text-left mt-8">{faqData.map((faq, index) => (<FaqItem key={index} question={faq.q} answer={faq.a} isOpen={openFaq === index} onClick={() => setOpenFaq(openFaq === index ? null : index)}/>))}</div></div></section><section className="py-20 px-4 bg-gray-900"><div className="max-w-4xl mx-auto"><div className="relative bg-gradient-to-r from-purple-600/20 to-sky-500/20 rounded-2xl p-8 md:p-12 shadow-2xl overflow-hidden border border-gray-700"><div className="relative z-10 text-center"><h2 className="text-4xl font-bold mb-3">Tetap Terhubung!</h2><p className="text-gray-300 max-w-lg mx-auto mb-8">Berlangganan buletin kami untuk mendapatkan pembaruan, acara, dan peluang terbaru.</p><form onSubmit={handleSubscribe} className="flex flex-col sm:flex-row gap-4 max-w-md mx-auto"><input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Masukkan email Anda" className="w-full px-5 py-3 rounded-lg bg-gray-800/50 border border-gray-700 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all" required/><button type="submit" className="flex-shrink-0 px-6 py-3 bg-purple-600 text-white font-semibold rounded-lg hover:bg-purple-700 transition-colors flex items-center justify-center gap-2">Subscribe<svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" /></svg></button></form></div></div></div></section><footer className="text-center py-8 text-gray-500 border-t border-gray-800">Dibuat dengan 💜 oleh Anda.</footer></main>);
};

// --- Komponen Aplikasi Utama ---
export default function App() {
  const [currentPage, setCurrentPage] = useState('landing'); // 'landing', 'types', 'detail', 'classifier'
  const [selectedCategory, setSelectedCategory] = useState(null);

  const handleSelectCategory = (category) => {
      setSelectedCategory(category);
      setCurrentPage('detail');
  };

  const renderPage = () => {
      switch(currentPage) {
          case 'types':
              return <WasteTypesPage onSelectCategory={handleSelectCategory} onBack={() => setCurrentPage('landing')} onGoToClassifier={() => setCurrentPage('classifier')} />;
          case 'detail':
              return <WasteDetailPage category={selectedCategory} onBack={() => setCurrentPage('types')} />;
          case 'classifier':
              return <ClassifierPage onBack={() => setCurrentPage('types')} />;
          case 'landing':
          default:
              return <LandingPage onStart={() => setCurrentPage('types')} />;
      }
  };

  return (
    <div>
      {renderPage()}
      <a href="https://aksara.ai/" target="_blank" rel="noopener noreferrer" className="fixed bottom-6 right-6 bg-purple-600 text-white p-4 rounded-full shadow-lg hover:bg-purple-700 transform hover:scale-110 transition-all duration-300 z-50 flex items-center group hover:pr-6" title="Kunjungi Aksara AI"><AksaraIcon /><span className="max-w-0 overflow-hidden group-hover:max-w-48 transition-all duration-500 ease-in-out ml-0 group-hover:ml-3 whitespace-nowrap text-sm font-medium">Coba tanyakan ke Aksara AI!</span></a>
    </div>
  );
}
