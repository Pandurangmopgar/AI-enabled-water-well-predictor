function SignedInRoutes({ setIsSignedIn, isSignedIn }) {
  const navigate = useNavigate();

  const onSuccess = (response) => {
    console.log(response);
    setIsSignedIn(true);
    setTimeout(() => {
      navigate('/profile');
    }, 1000);
  };

  const onFailure = (error) => {
    console.log('Sign-in failed:', error);
  };

  return (
    <Routes>
      <Route path="/signin" element={<GoogleLogin clientId={CLIENT_ID} onSuccess={onSuccess} onFailure={onFailure} />} />
      <Route
        path="/profile"
        element={
          isSignedIn ? (
            <ChatAppContainer>
              <ChatApp />
            </ChatAppContainer>
          ) : (
            <div>Please sign in to access the chat app.</div>
          )
        }
      />
    </Routes>
  );
}