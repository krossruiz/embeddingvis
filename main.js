import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { pipeline, env } from '@xenova/transformers';
import T_SNE from 'tsne-js';
import { paragraphs } from './data.js';

// Skip local model checks since we are using CDN/NPM
env.allowLocalModels = false;
env.useBrowserCache = false;

// Configuration
const MODEL_NAME = 'Xenova/all-MiniLM-L6-v2';
const TSNE_ITERATIONS = 500;
const TSNE_PERPLEXITY = 10;
const TSNE_LEARNING_RATE = 100;

// DOM Elements
const statusEl = document.getElementById('status');
const tooltipEl = document.getElementById('tooltip');
const infoEl = document.getElementById('info');

// State
let embeddings = [];
let points = []; // Three.js meshes
let tsne;
let scene, camera, renderer, controls, raycaster, mouse;

async function init() {
    initThree();

    try {
        statusEl.innerText = 'Loading model...';
        const extractor = await pipeline('feature-extraction', MODEL_NAME);

        statusEl.innerText = 'Generating embeddings...';
        // Generate embeddings sequentially to avoid freezing UI too much
        const vectors = [];

        for (let i = 0; i < paragraphs.length; i++) {
            statusEl.innerText = `Embedding ${i + 1}/${paragraphs.length}...`;
            const output = await extractor(paragraphs[i], { pooling: 'mean', normalize: true });
            vectors.push(Array.from(output.data));
            // Small delay to let UI update
            await new Promise(r => setTimeout(r, 10));
        }

        embeddings = vectors;
        startTSNE();

    } catch (e) {
        statusEl.innerText = 'Error: ' + e.message;
        console.error(e);
    }
}

function initThree() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    // Add some fog for depth
    scene.fog = new THREE.FogExp2(0x000000, 0.02);

    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 20;

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);

    // Raycaster
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    window.addEventListener('resize', onWindowResize, false);
    window.addEventListener('mousemove', onMouseMove, false);

    animate();
}

function startTSNE() {
    statusEl.innerText = 'Initializing T-SNE...';

    try {
        // Initialize T-SNE
        // Based on example: new TSNE({ ... })
        // And it seems T_SNE from import is the class (based on previous log {default: class})
        // But let's be safe and check default
        const TSNE_Class = T_SNE.default || T_SNE;

        tsne = new TSNE_Class({
            epsilon: TSNE_LEARNING_RATE,
            perplexity: TSNE_PERPLEXITY,
            dim: 3,
            nIter: TSNE_ITERATIONS
        });

        // Setup events
        tsne.on('progressIter', (iter) => {
            statusEl.innerText = `Projecting T-SNE: ${iter}/${TSNE_ITERATIONS}`;
        });

        tsne.on('progressData', (pos) => {
            updatePoints(pos);
        });

        tsne.on('progressStatus', (status) => {
            console.log('TSNE Status:', status);
        });

        tsne.on('done', (pos) => {
            statusEl.innerText = 'Done!';
            updatePoints(pos);
        });

        // Initialize data
        tsne.init({
            data: embeddings,
            type: 'dense'
        });

        // Create points in Three.js at random positions initially
        const geometry = new THREE.SphereGeometry(0.2, 16, 16);

        paragraphs.forEach((text, i) => {
            const material = new THREE.MeshBasicMaterial({
                color: new THREE.Color().setHSL(Math.random(), 0.7, 0.5)
            });
            const sphere = new THREE.Mesh(geometry, material);

            // Random initial position
            sphere.position.set(
                (Math.random() - 0.5) * 10,
                (Math.random() - 0.5) * 10,
                (Math.random() - 0.5) * 10
            );

            sphere.userData = { text: text, id: i };
            scene.add(sphere);
            points.push(sphere);
        });

        // Run T-SNE (async/event-based)
        tsne.run();

    } catch (e) {
        console.error("Failed to init tSNE:", e);
        statusEl.innerText = 'Error initializing T-SNE: ' + e.message;
    }
}

function updatePoints(pos) {
    if (!pos || pos.length === 0) return;

    // Find bounds to normalize/scale
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    pos.forEach(p => {
        minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
        minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
        minZ = Math.min(minZ, p[2]); maxZ = Math.max(maxZ, p[2]);
    });

    const scale = 15; // Desired spread size

    points.forEach((point, i) => {
        const p = pos[i];
        if (!p) return;

        // Normalize to -0.5 to 0.5 then scale
        const x = ((p[0] - minX) / (maxX - minX || 1) - 0.5) * scale;
        const y = ((p[1] - minY) / (maxY - minY || 1) - 0.5) * scale;
        const z = ((p[2] - minZ) / (maxZ - minZ || 1) - 0.5) * scale;

        // Lerp for smooth movement
        point.position.lerp(new THREE.Vector3(x, y, z), 0.1);
    });
}

function onMouseMove(event) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    // Update tooltip position
    tooltipEl.style.left = event.clientX + 15 + 'px';
    tooltipEl.style.top = event.clientY + 15 + 'px';
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();

    // Raycasting
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(points);

    if (intersects.length > 0) {
        const object = intersects[0].object;
        const text = object.userData.text;

        // Highlight
        if (window.hoveredObject !== object) {
            if (window.hoveredObject) window.hoveredObject.scale.set(1, 1, 1);
            window.hoveredObject = object;
            object.scale.set(1.5, 1.5, 1.5);

            tooltipEl.style.display = 'block';
            tooltipEl.innerText = text;
        }
    } else {
        if (window.hoveredObject) {
            window.hoveredObject.scale.set(1, 1, 1);
            window.hoveredObject = null;
            tooltipEl.style.display = 'none';
        }
    }

    renderer.render(scene, camera);
}

init();
