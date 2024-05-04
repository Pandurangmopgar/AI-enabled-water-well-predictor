// GoogleSignInPage.js
import React from 'react';
import { useGoogleLogin } from '@react-oauth/google';
import axios from 'axios';
import { useHistory } from 'react-router-dom';

function GoogleSignInPage() {
  const history = useHistory();
  const login = useGoogleLogin({
    onSuccess: (codeResponse) => {
      // Handle successful sign-in
      axios
        .get(
          `https://www.googleapis.com/oauth2/v1/userinfo?access_token=${codeResponse.access_token}`,
          {
            headers: {
              Authorization: `Bearer ${codeResponse.access_token}`,
              Accept: 'application/json',
            },
          }
        )
        .then((res) => {
          // Store user profile data or perform other actions
          console.log(res.data);
          // Redirect to the profile page after successful sign-in
          history.push('/profile');
        })
        .catch((err) => console.log(err));
    },
    onError: (error) => console.log('Login Failed:', error),
  });

  return (
    <div>
      <h2>Sign in with Google</h2>
      <button onClick={login}>Sign in with Google 🚀</button>
    </div>
  );
}

export default GoogleSignInPage;