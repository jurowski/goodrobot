# Developer Guide

## Development Environment Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- npm 9+

### Quick Start

1. Clone the repository and set up the environment:
```bash
git clone <repository-url>
cd goodrobot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Install frontend dependencies:
```bash
cd docs-site
npm install
cd ..
```

3. Start the development environment:
```bash
# Make the script executable (only needed once)
chmod +x scripts/start_dev.sh

# Start both servers with a single command
./scripts/start_dev.sh
```

This will start both the Next.js frontend and FastAPI backend with hot-reloading enabled. The script:
- Automatically kills any existing processes on ports 3000 and 8000
- Starts both servers with proper configuration
- Handles cleanup when you press Ctrl+C

## Documentation Site Development

### Project Structure
```
docs-site/
├── src/
│   ├── components/           # React components
│   │   ├── Search.tsx        # Main search component
│   │   └── CodePreview.tsx   # Code preview component
│   ├── pages/                # Next.js pages
│   │   ├── index.tsx         # Main search page
│   │   └── system/[id].tsx   # System documentation
│   ├── context/              # React context
│   ├── services/             # API services
│   └── utils/                # Utility functions
└── public/                   # Static assets
```

### Key Components

#### Search Component
The main search functionality is implemented in `src/components/Search.tsx`. This component:
- Provides real-time search across all documentation
- Supports category filtering
- Highlights matching text in search results
- Handles loading and error states

#### Routing
- `/` - Main documentation search page
- `/system/[id]` - Individual system documentation
- All other routes are handled by Next.js's file-based routing

### Development Workflow

1. **Making Changes to Search**
   - Modify `src/components/Search.tsx` for search functionality
   - Update `src/styles/SearchResults.module.css` for styling
   - Test changes in the browser at `http://localhost:3000`

2. **Adding Documentation**
   - Add new system documentation in the appropriate format
   - Update the documentation context in `src/context/DocumentationContext.tsx`
   - Test search functionality with new content

3. **Styling Updates**
   - Modify CSS modules in `src/styles/`
   - Use CSS modules for component-specific styles
   - Test responsive design across different screen sizes

## Development Servers

The project uses two development servers that are managed by the `start_dev.sh` script:

### FastAPI Backend
- Runs on http://localhost:8000
- Features:
  - Automatic reload on file changes
  - Interactive API documentation at `/docs`
  - Real-time error reporting
  - Hot reload for Python code

### Next.js Frontend
- Runs on http://localhost:3000
- Features:
  - Hot module replacement
  - Fast refresh for React components
  - Development error overlay
  - Source maps for debugging

## Development Workflow

### Starting Development
1. Run `./scripts/start_dev.sh` to start both servers
2. Make changes to the code
3. Servers will automatically reload with changes
4. Test changes in the browser

### Stopping Development
- Press `Ctrl+C` to stop both servers gracefully
- The script will clean up all processes

### Troubleshooting

#### Port Conflicts
If you see port conflicts, the `start_dev.sh` script will automatically handle them by:
1. Killing any existing processes on ports 3000 and 8000
2. Starting fresh instances of both servers

If you need to manually kill processes:
```bash
# Kill processes on ports 3000 and 8000
lsof -ti:3000,8000 | xargs kill -9
```

#### File Watching Issues
If file watching stops working:
1. Stop the servers with `Ctrl+C`
2. Run `./scripts/start_dev.sh` again

#### Common Errors
- **Module Import Errors**: Make sure you're running from the project root
- **Port Already in Use**: The script will handle this automatically
- **Hot Reload Not Working**: Check file permissions and try restarting

## Best Practices

1. **Code Changes**
   - Make small, focused changes
   - Test changes immediately
   - Use the development servers for rapid feedback

2. **File Watching**
   - Save files frequently
   - Watch the terminal for reload messages
   - Check browser console for errors

3. **Debugging**
   - Use browser dev tools for frontend issues
   - Check terminal output for backend errors
   - Use the interactive API docs for testing endpoints

## Advanced Configuration

### Customizing File Watching
The development servers use:
- FastAPI's `reload=True` for backend watching
- Next.js's built-in file watching for frontend

### Environment Variables
Create a `.env` file in the project root:
```env
# Backend
API_HOST=0.0.0.0
API_PORT=8000

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Test using the development servers
4. Submit a pull request

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [Project Structure](#project-structure)
- [API Documentation](API.md)

## Browser Testing and Configuration

### Recommended Browsers for Development

For development and testing, we recommend using Firefox for the following reasons:

1. **WebSocket Support**: Firefox has excellent WebSocket support and is more forgiving with development environments
2. **Clean Environment Separation**: Better separation between development and production environments
3. **Less Aggressive Caching**: More predictable caching behavior compared to Chrome
4. **Development Tools**: Comprehensive set of development tools and features

### Firefox Development Configuration

To optimize Firefox for development:

1. Open Firefox and go to `about:config`
2. Set the following preferences:
   ```javascript
   // Allow insecure WebSocket connections during development
   network.websocket.allowInsecureFromHTTPS = true

   // Disable cache for development
   browser.cache.disk.enable = false
   browser.cache.memory.enable = false

   // Enable development tools
   devtools.debugger.remote-enabled = true
   ```

3. Install the following recommended extensions:
   - React Developer Tools
   - Redux DevTools
   - Web Developer Tools

### Browser Cache Management

To prevent caching issues during development:

1. **Clear Cache**: Use `Ctrl+Shift+Del` (Windows/Linux) or `Cmd+Shift+Del` (Mac) to clear browser cache
2. **Private Browsing**: Use private/incognito windows for testing
3. **Cache Control Headers**: The development server is configured to send appropriate cache control headers

### Development Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Development Environment
ENVIRONMENT=development

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Development Features
DISABLE_HTTPS_REDIRECT=true
ENABLE_DEV_TOOLS=true
```

### Troubleshooting Browser Issues

If you encounter issues:

1. **HTTPS Redirect Problems**:
   - Clear browser cache
   - Use private browsing window
   - Verify `DISABLE_HTTPS_REDIRECT=true` in `.env`

2. **WebSocket Connection Issues**:
   - Check Firefox WebSocket settings
   - Verify `network.websocket.allowInsecureFromHTTPS = true`
   - Ensure correct WebSocket URL in frontend configuration

3. **Cache-Related Issues**:
   - Clear browser cache
   - Use private browsing window
   - Disable browser cache in development

## Voice Interface Components

### Activity Log
The voice interface includes an activity log panel that displays system events and messages:
- Real-time logging of client and server events
- Filterable by log level (DEBUG, INFO, WARNING, ERROR)
- Search functionality
- Copy to clipboard feature
- Clear logs option
- Collapsible panel design

### Audio Visualization
The interface provides visual feedback for audio input:
- Microphone icon with blue glow effect when active
- Animated wave rings indicating audio detection
- Volume level meter with real-time feedback
- Responsive design that scales with window size

### Styling Variables
The interface uses CSS variables for consistent theming:
```css
:root {
    --primary-color: #3498db;
    --secondary-color: #2c3e50;
    --background-color: #1a1a1a;
    --surface-color: #2d2d2d;
    --text-color: #e0e0e0;
    --border-color: #404040;
    --hover-color: #2980b9;
    --success-color: #4CAF50;
    --error-color: #ff6b6b;
    --muted-text: #a0a0a0;
}
```

### Responsive Design
The interface is fully responsive:
- Adapts to different screen sizes
- Column collapsing for space efficiency
- Fluid typography and spacing
- Mobile-friendly controls

// ... existing code ...
