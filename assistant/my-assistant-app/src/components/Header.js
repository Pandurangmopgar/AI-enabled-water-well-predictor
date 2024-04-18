import React from 'react';
import styled from 'styled-components';

const Header = () => {
  return (
    <HeaderContainer>
      <Title>Conversational Assistant</Title>
      <UserAvatar>
        <img src="user-avatar.png" alt="User Avatar" />
      </UserAvatar>
    </HeaderContainer>
  );
};

export default Header;

const HeaderContainer = styled.header`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  background-color: #007bff;
  color: #fff;
`;

const Title = styled.h1`
  font-size: 1.5rem;
  font-weight: bold;
`;

const UserAvatar = styled.div`
  width: 40px;
  height: 40px;
  border-radius: 50%;
  overflow: hidden;
  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
`;