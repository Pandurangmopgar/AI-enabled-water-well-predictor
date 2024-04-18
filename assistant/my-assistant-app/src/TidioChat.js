import React, { useEffect } from 'react';

const TidioChat = () => {
  useEffect(() => {
    // Add the script tag to the document
    const script = document.createElement('script');
    script.src = '//code.tidio.co/6d50f5g0lwp9lcztaqgedolinzgl.js';
    script.async = true;
    document.body.appendChild(script);

    // Clean up the script when the component is unmounted
    return () => {
      document.body.removeChild(script);
    };
  }, []);

  return null; // This component doesn't render any UI
};

export default TidioChat;
