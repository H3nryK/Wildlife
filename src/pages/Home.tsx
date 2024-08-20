import React from 'react';
import { Link } from 'react-router-dom';
import { HeartIcon } from '@heroicons/react/solid';

const Home = () => {
  return (
    <div className="bg-gradient-to-r from-pink-200 to-yellow-100 min-h-screen flex items-center justify-center p-6">
      <div className="bg-white p-10 rounded-lg shadow-lg max-w-md mx-auto text-center">
        <div className="mb-4">
          <HeartIcon className="h-16 w-16 text-pink-500 mx-auto animate-bounce" />
        </div>
        <h2 className="text-3xl font-bold text-pink-600 mb-4 font-poppins">
          A Special Note for You
        </h2>
        <p className="text-gray-700 mb-8 font-poppins text-lg">
          Hey [Your Crush's Name], I wanted to express how much you mean to me. You light up my life, and I can't help but smile whenever I think of you. These songs make me think of you, and I hope they make you smile too! 😊
        </p>
        <Link to="/playlist">
          <button className="bg-pink-500 text-white py-3 px-6 rounded-full hover:bg-pink-600 transition duration-300 focus:outline-none focus:ring font-poppins">
            Listen to my playlist
          </button>
        </Link>
      </div>
    </div>
  );
};

export default Home;
