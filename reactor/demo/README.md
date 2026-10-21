# Streaming CWM browser demo

Prepare `public/examples/` as described in the [model guide](../README.md), then:

```sh
cp .env.example .env
npm ci
npm run dev
```

Open http://localhost:3000. Set **Runtime** to the HTTP endpoint reached by your browser, click **Connect**, choose an example or upload conditions, an opening image and a prompt, then click **Start rendering**. The example selector changes all three inputs together. The condition MP4 is a visual preview; lossless depth and semantic-ID pairs drive inference. Prompt and reference stay fixed for a render. Use **Stop rendering** before changing them.

Reference video is on the left and generated video on the right. The service publishes matching semantic and generated frames together on separate video tracks at 24 FPS. Browser buffering may cause a brief offset between them. The appearance-image preview sits below upload **02**. This layout works for official examples and custom condition ZIPs; it does not depend on an independently playing preview MP4. Rebuild Runtime after changing the adapter so the reference track is available. An older Runtime still displays generated video.

The **Runtime** address is editable while disconnected. To switch ports, click **Disconnect**, change the address, then **Connect**. A link such as `http://localhost:3000/?endpoint=http%3A%2F%2Flocalhost%3A8080` prefills the address without connecting. It takes precedence over `VITE_REACTOR_URL`, whose default is `http://localhost:8080`. Changing this build-time default requires restarting Vite. The provided client uses Reactor's local SDK mode. A hosted service requires a server-side authentication exchange; never put a Reactor API key into a `VITE_*` variable.

For SSH access, forward the frontend, Runtime HTTP and authenticated TURN/TCP ports configured by your server operator. An example, with placeholder ports to replace:

```sh
ssh -N -L 3000:127.0.0.1:3000 -L 8080:127.0.0.1:8080 \
  -L 3478:127.0.0.1:3478 user@gpu-host
```

This command assumes TURN/TCP is already configured and its advertised relay addresses are reachable from the model container. It does not create a relay. HTTP forwarding and Docker host networking alone do not forward WebRTC UDP.

```sh
npm test
npm run build
npm run preview
```

The Vite configuration preserves the official SDK's WebAssembly assets in development and production. Selecting inputs does not start inference; command failures remain visible. On a disconnect, wait for the previous session to finish closing before reconnecting.
