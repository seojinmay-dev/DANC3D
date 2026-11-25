import {defineConfig,loadEnv} from 'vite';
export default defineConfig(({mode})=>{
  const env=loadEnv(mode,process.cwd(),'');
  const target=env.DANC3D_API_TARGET||'http://127.0.0.1:8000';
  const proxy={'/ws':{target,ws:true},'/api':{target:env.DANC3D_API_TARGET||'http://127.0.0.1:8000',changeOrigin:true,rewrite:path=>path.replace(/^\/api(?=\/|$)/,'')}};
  // Keep the browser origin aligned with the backend's local allowlist.
  // A second launch must report a port conflict instead of silently moving to 5174.
  const localServer={port:5173,strictPort:true,proxy};
  return {esbuild:{jsx:'automatic'},server:localServer,preview:{...localServer}};
});
