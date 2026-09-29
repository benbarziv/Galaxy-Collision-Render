# Galaxy Collision Simulator

A browser app that simulates two spiral galaxies colliding. The stars move under gravity in real time. Nothing is scripted.

## Run it

```bash
npm install
npm run dev
```

Open [http://localhost:5174](http://localhost:5174).

Drag to look around. Scroll to zoom. Use the panel on the left to change the collision, speed, and quality.

## Scenarios

- **Head-on Collision** — two similar galaxies meet straight on and merge.
- **Grazing Collision** — they slide past each other and stretch out long tails.
- **High-speed Flyby** — they pass too fast to merge. The disks get pulled out of shape, then they leave.
- **Unequal-mass Merger** — a smaller galaxy falls into a bigger one.

Turn on **Stats** if you want the numbers: simulated time, separation, speed, and how many stars have left the box.
