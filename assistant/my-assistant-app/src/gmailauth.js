import React from 'react';
import { GoogleLogin,  googleLogout} from '@react-oauth/google';

const CLIENT_ID = '658843026604-vf7nufcr77vnm3r06574nes4faiejghg.apps.googleusercontent.com'; // replace with your client ID

const GmailAuth = () => {
  const [user, setUser] = React.useState(null);

  const responseGoogle = (response) => {
    console.log(response);
    setUser(response.profileObj);
  };

  const logout = () => {
    setUser(null);
  };

  return (
    <div>
      {user ? (
        <googleLogout
          clientId={CLIENT_ID}
          buttonText="Logout"
          onLogoutSuccess={logout}
        />
      ) : (
        <GoogleLogin
          clientId={CLIENT_ID}
          buttonText="Login"
          onSuccess={responseGoogle}
          onFailure={responseGoogle}
          cookiePolicy={'single_host_origin'}
        />
      )}
    </div>
  );
};

export default GmailAuth;