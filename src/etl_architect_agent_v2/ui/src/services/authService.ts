// Simple authentication service
// In a real application, this would integrate with your authentication system
// For now, we'll use a mock implementation

let currentUserId: string | null = null;

export const getUserId = (): string => {
  return 'test_user';
};

export const setUserId = (userId: string): void => {
  currentUserId = userId;
};

export const clearUserId = (): void => {
  currentUserId = null;
};

export const isAuthenticated = (): boolean => {
  return true;
};

// Mock authentication functions for development
export const login = async (username: string, password: string): Promise<void> => {
  // In a real application, this would make an API call to authenticate
  console.log('Login:', username);
};

export const logout = async (): Promise<void> => {
  // In a real application, this would clear the auth token
  console.log('Logout');
}; 