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
        Hey Judith! 😊
        </p>
        <p className="text-gray-700 mb-8 font-poppins text-lg">
          I’ve been thinking about how lucky I am to have someone like you in my life—smart, driven, and just all-around amazing! 🤓 We both rock those glasses, but I have to say, you take it to another level with your tech-savvy brilliance. 💻✨ Every time I see you dive into a project or tackle a challenge, I'm reminded of how unstoppable you are.
        </p>
        <p className="text-gray-700 mb-8 font-poppins text-lg">
          But I want to remind you of something important: you can be anything you set your mind to, no matter where you come from or what you’ve been through. 🌟 Life can throw some pretty crazy curveballs, but I believe in you. You’ve got the brains, the talent, and the heart to overcome anything. So, keep your head up, stay strong, and always reach for the stars. 🌠
        </p>
        <p className="text-gray-700 mb-8 font-poppins text-lg">
          And don’t forget to have a little fun along the way! 😄 You’re one of those rare people who can make even the toughest situations feel a little lighter, and that’s a gift. So, keep spreading those good vibes, and keep being your awesome self. The world needs more people like you—bold, smart, and unafraid to be themselves. 🌍💡
        </p>
        <p className="text-gray-700 mb-8 font-poppins text-lg">
          I'm so excited to see where your journey takes you. Whether it’s conquering the tech world or just owning every day with that unstoppable energy of yours, I know you’re going to do incredible things. 🚀
        </p>
        <p className="text-gray-700 mb-8 font-poppins text-lg">
          Stay fabulous, keep striving, and never forget—you’ve got a friend right here cheering you on every step of the way. 😎💖
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
