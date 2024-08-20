import React, { useEffect, useState } from 'react';

const Preloader = () => {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setLoading(false), 2000); // 2 seconds delay
    return () => clearTimeout(timer);
  }, []);

  return (
    loading ? (
      <div className="fixed inset-0 bg-pink-200 flex items-center justify-center">
        <div className="text-pink-600 text-xl animate-spin">💖</div>
      </div>
    ) : null
  );
};

export default Preloader;
