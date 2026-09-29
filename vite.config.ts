import { defineConfig } from 'vite';

export default defineConfig({
  // Relative base so the built bundle works from the filesystem, a sub-path, or
  // any static host without rewriting asset URLs.
  base: './',
  server: {
    // Single source of truth for the port. The dev script and the docs both
    // refer to 5174, and declaring it here as well as on the command line meant
    // the two could drift: bare `vite` would listen on 5173 while `npm run dev`
    // listened on 5174, so which URL you opened decided whether the app was
    // there at all.
    port: 5174,
    strictPort: true,
    open: false,
    // Bind explicitly instead of letting Node pick.
    //
    // Vite's default host is `localhost`, which on current macOS resolves to the
    // IPv6 loopback first. The server then listens on [::1] only, and anything
    // that resolves to 127.0.0.1 -- including the test scripts' default URL and
    // any browser resolving via IPv4 -- gets ERR_CONNECTION_REFUSED. That looks
    // exactly like "the server died", and it is a confusing way to lose an
    // afternoon: `curl localhost:5174` works while `curl 127.0.0.1:5174` does
    // not, in the same shell, at the same moment.
    //
    // `true` listens on all interfaces, so both loopback addresses work and a
    // device on the same network can reach it too.
    host: true,
  },
  preview: { port: 4173, strictPort: true },
  build: { target: 'es2020', outDir: 'dist', sourcemap: true },
});
