import React from 'react';
import { MusicNoteIcon } from '@heroicons/react/solid';

const PlaylistPage = () => {
  const videos = [
    { title: 'Song 1', url: 'https://www.youtube.com/embed/song1' },
    { title: 'Song 2', url: 'https://www.youtube.com/embed/song2' },
    { title: 'Song 3', url: 'https://www.youtube.com/embed/song3' },
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
              height="200"
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
