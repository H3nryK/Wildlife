import { MusicNoteIcon } from '@heroicons/react/solid';

const PlaylistPage = () => {
  const videos = [
    { title: 'Johnny Drille - Count My Blessings', url: "https://www.youtube.com/embed/ZAwQ-TJFEO8?list=RDZAwQ-TJFEO8" },
    { title: 'Lil Uzi Vert - The Way Life Goes Remix (Feat. Nicki Minaj)', url: 'https://www.youtube.com/embed/SxAp27sFaIM?list=RDGMEMHDXYb1_DDSgDsobPsOFxpA' },
    { title: 'Ayra Starr - Rush', url: "https://www.youtube.com/embed/crtQSTYWtqE?list=RDGMEMHDXYb1_DDSgDsobPsOFxpA" },
  ];

  return (
    <div className="bg-gradient-to-r from-yellow-100 to-pink-200 min-h-screen p-10">
      <h1 className="text-4xl font-bold text-center text-pink-600 mb-8 font-poppins flex items-center justify-center">
        <MusicNoteIcon className="h-10 w-10 text-pink-500 mr-2" />
        Our Playlist
      </h1>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {videos.map((video, index) => (
          <div key={index} className="bg-white p-6 rounded-lg shadow-lg hover:shadow-xl transition-shadow duration-300">
            <h3 className="text-lg font-semibold mb-2 font-poppins">{video.title}</h3>
            <iframe
              width="100%"
              height="300"
              src={video.url}
              title={video.title}
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            ></iframe>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PlaylistPage;
