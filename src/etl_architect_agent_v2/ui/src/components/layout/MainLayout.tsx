import React from 'react';
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  useTheme,
  useMediaQuery,
  styled
} from '@mui/material';
import {
  Menu as MenuIcon,
  Chat as ChatIcon,
  DataObject as DataIcon,
  Assessment as QualityIcon,
  CloudUpload as UploadIcon
} from '@mui/icons-material';
import { Sidebar } from './Sidebar';
import { ChatPanel } from './ChatPanel';

const drawerWidth = 280;
const chatPanelWidth = 320;

const Main = styled('main', { shouldForwardProp: (prop) => prop !== 'open' })<{
  open?: boolean;
  chatOpen?: boolean;
}>(({ theme, open, chatOpen }) => ({
  flexGrow: 1,
  padding: theme.spacing(3),
  transition: theme.transitions.create('margin', {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  marginLeft: `-${drawerWidth}px`,
  marginRight: chatOpen ? `-${chatPanelWidth}px` : 0,
  ...(open && {
    transition: theme.transitions.create('margin', {
      easing: theme.transitions.easing.easeOut,
      duration: theme.transitions.duration.enteringScreen,
    }),
    marginLeft: 0,
  }),
}));

interface MainLayoutProps {
  children: React.ReactNode;
}

export const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [sidebarOpen, setSidebarOpen] = React.useState(!isMobile);
  const [chatOpen, setChatOpen] = React.useState(false);

  const handleDrawerToggle = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const handleChatToggle = () => {
    setChatOpen(!chatOpen);
  };

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      <AppBar
        position="fixed"
        sx={{
          zIndex: theme.zIndex.drawer + 1,
          backgroundColor: 'background.paper',
          color: 'text.primary',
          boxShadow: 1
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            onClick={handleDrawerToggle}
            edge="start"
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            Data Agent
          </Typography>
          <IconButton color="inherit" onClick={handleChatToggle}>
            <ChatIcon />
          </IconButton>
        </Toolbar>
      </AppBar>

      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        variant={isMobile ? 'temporary' : 'permanent'}
        width={drawerWidth}
      />

      <Main open={sidebarOpen} chatOpen={chatOpen}>
        <Toolbar /> {/* Spacer for AppBar */}
        {children}
      </Main>

      <ChatPanel
        open={chatOpen}
        onClose={() => setChatOpen(false)}
        width={chatPanelWidth}
      />
    </Box>
  );
}; 