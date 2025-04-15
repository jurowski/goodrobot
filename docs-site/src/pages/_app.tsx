import { AppProps } from 'next/app';
import { DocumentationProvider } from '../context/DocumentationContext';

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <DocumentationProvider>
      <Component {...pageProps} />
    </DocumentationProvider>
  );
}

export default MyApp;
