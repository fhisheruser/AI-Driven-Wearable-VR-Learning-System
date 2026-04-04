/**
 * Three.js 3D Renderer for the AI-Driven VR Learning System.
 *
 * Renders scene descriptions from the visualization engine as
 * interactive 3D visualizations using Three.js.
 */

class SceneRenderer {
    constructor(canvasId, containerId) {
        this.canvas = document.getElementById(canvasId);
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.objects = new Map();
        this.labels = [];
        this.animationId = null;
        this.autoRotate = false;
        this.showLabels = true;
        this.clock = new THREE.Clock();
        this.mouse = new THREE.Vector2();
        this.raycaster = new THREE.Raycaster();
        this.hoveredObject = null;
        this.isDragging = false;
        this.previousMouse = { x: 0, y: 0 };
        this.cameraDistance = 5;
        this.cameraAngleX = 0;
        this.cameraAngleY = 0.3;

        this._init();
    }

    _init() {
        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true,
        });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setClearColor(0x000011);
        this.renderer.shadowMap.enabled = true;

        // Scene
        this.scene = new THREE.Scene();

        // Camera
        this.camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
        this.camera.position.set(0, 2, 5);

        // Default lighting
        this._setupLighting();

        // Events
        this._setupEvents();

        // Resize
        this._resize();

        // Start render loop
        this._animate();
    }

    _setupLighting(config) {
        // Remove existing lights
        this.scene.children
            .filter(c => c.isLight)
            .forEach(l => this.scene.remove(l));

        const ambient = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambient);

        const directional = new THREE.DirectionalLight(0xffffff, 0.8);
        directional.position.set(5, 10, 5);
        directional.castShadow = true;
        this.scene.add(directional);

        const hemisphere = new THREE.HemisphereLight(0x87CEEB, 0x362d2d, 0.3);
        this.scene.add(hemisphere);
    }

    _setupEvents() {
        window.addEventListener('resize', () => this._resize());

        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.previousMouse.x = e.clientX;
            this.previousMouse.y = e.clientY;
        });

        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

            if (this.isDragging) {
                const dx = e.clientX - this.previousMouse.x;
                const dy = e.clientY - this.previousMouse.y;
                this.cameraAngleX -= dx * 0.005;
                this.cameraAngleY = Math.max(-1.5, Math.min(1.5,
                    this.cameraAngleY - dy * 0.005));
                this._updateCameraPosition();
                this.previousMouse.x = e.clientX;
                this.previousMouse.y = e.clientY;
            }
        });

        this.canvas.addEventListener('mouseup', () => { this.isDragging = false; });
        this.canvas.addEventListener('mouseleave', () => { this.isDragging = false; });

        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.cameraDistance = Math.max(2, Math.min(30,
                this.cameraDistance + e.deltaY * 0.01));
            this._updateCameraPosition();
        }, { passive: false });
    }

    _updateCameraPosition() {
        const x = this.cameraDistance * Math.sin(this.cameraAngleX) * Math.cos(this.cameraAngleY);
        const y = this.cameraDistance * Math.sin(this.cameraAngleY);
        const z = this.cameraDistance * Math.cos(this.cameraAngleX) * Math.cos(this.cameraAngleY);
        this.camera.position.set(x, y, z);
        this.camera.lookAt(this.cameraTarget || new THREE.Vector3(0, 0, 0));
    }

    _resize() {
        const rect = this.container.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;
        this.renderer.setSize(w, h);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
    }

    /**
     * Render a scene from the visualization engine output.
     * @param {Object} sceneData - Scene data from the API
     */
    renderScene(sceneData) {
        // Clear existing scene objects
        this._clearScene();

        if (!sceneData || !sceneData.objects) return;

        // Set background
        if (sceneData.background_color) {
            this.renderer.setClearColor(new THREE.Color(sceneData.background_color));
        }

        // Set camera
        if (sceneData.camera_position) {
            const cp = sceneData.camera_position;
            this.cameraDistance = Math.sqrt(cp[0]**2 + cp[1]**2 + cp[2]**2);
            this.cameraAngleX = Math.atan2(cp[0], cp[2]);
            this.cameraAngleY = Math.asin(cp[1] / this.cameraDistance);
            this.camera.position.set(cp[0], cp[1], cp[2]);
        }
        if (sceneData.camera_target) {
            const ct = sceneData.camera_target;
            this.cameraTarget = new THREE.Vector3(ct[0], ct[1], ct[2]);
            this.camera.lookAt(this.cameraTarget);
        }

        // Set lighting
        if (sceneData.lighting) {
            this._setupLighting(sceneData.lighting);
        }

        // Create objects
        for (const objData of sceneData.objects) {
            const mesh = this._createObject(objData);
            if (mesh) {
                this.scene.add(mesh);
                this.objects.set(objData.id, { mesh, data: objData });
            }
        }

        // Show placeholder hide
        const placeholder = document.getElementById('viz-placeholder');
        if (placeholder) placeholder.style.display = 'none';

        // Add annotations
        this._renderAnnotations(sceneData.annotations || []);
    }

    _createObject(data) {
        let geometry, material, mesh;
        const color = new THREE.Color(data.color || '#ffffff');
        const isEmissive = data.metadata && data.metadata.emissive;

        material = new THREE.MeshPhongMaterial({
            color: color,
            transparent: data.opacity < 1.0,
            opacity: data.opacity || 1.0,
            shininess: 60,
            emissive: isEmissive ? color : new THREE.Color(0x000000),
            emissiveIntensity: isEmissive ? 0.5 : 0,
            side: data.opacity < 1.0 ? THREE.DoubleSide : THREE.FrontSide,
        });

        const s = data.scale || [1, 1, 1];

        switch (data.type) {
            case 'sphere':
                geometry = new THREE.SphereGeometry(s[0], 32, 32);
                mesh = new THREE.Mesh(geometry, material);
                break;

            case 'box':
                geometry = new THREE.BoxGeometry(s[0], s[1], s[2]);
                mesh = new THREE.Mesh(geometry, material);
                break;

            case 'cylinder':
                geometry = new THREE.CylinderGeometry(s[0], s[0], s[1], 16);
                mesh = new THREE.Mesh(geometry, material);
                break;

            case 'capsule':
                geometry = new THREE.CapsuleGeometry(s[1], s[0], 8, 16);
                mesh = new THREE.Mesh(geometry, material);
                break;

            case 'ring':
                geometry = new THREE.RingGeometry(s[0] - 0.02, s[0] + 0.02, 64);
                material.side = THREE.DoubleSide;
                mesh = new THREE.Mesh(geometry, material);
                mesh.rotation.x = Math.PI / 2;
                break;

            case 'arrow':
                const dir = new THREE.Vector3(1, 0, 0);
                mesh = new THREE.ArrowHelper(dir, new THREE.Vector3(0, 0, 0), s[1], color.getHex(), 0.3, 0.15);
                break;

            case 'custom':
            case 'text':
                // Use a sphere as fallback for custom types
                geometry = new THREE.SphereGeometry(Math.max(s[0], s[1], s[2]) * 0.5, 16, 16);
                mesh = new THREE.Mesh(geometry, material);
                break;

            default:
                geometry = new THREE.SphereGeometry(0.5, 16, 16);
                mesh = new THREE.Mesh(geometry, material);
        }

        if (mesh) {
            const pos = data.position || [0, 0, 0];
            mesh.position.set(pos[0], pos[1], pos[2]);

            if (data.rotation && !(mesh instanceof THREE.ArrowHelper)) {
                mesh.rotation.set(data.rotation[0], data.rotation[1], data.rotation[2]);
            }

            mesh.userData = {
                id: data.id,
                label: data.label,
                interactive: data.interactive,
                animation: data.animation,
            };

            // Add label sprite
            if (data.label && this.showLabels) {
                const label = this._createLabel(data.label, pos);
                this.scene.add(label);
                this.labels.push(label);
            }
        }

        return mesh;
    }

    _createLabel(text, position) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 256;
        canvas.height = 64;

        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.roundRect(0, 0, canvas.width, canvas.height, 8);
        ctx.fill();

        ctx.fillStyle = '#ffffff';
        ctx.font = '20px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, canvas.width / 2, canvas.height / 2);

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
        const sprite = new THREE.Sprite(material);
        sprite.position.set(position[0], position[1] + 0.8, position[2]);
        sprite.scale.set(1.5, 0.4, 1);
        sprite.userData.isLabel = true;

        return sprite;
    }

    _renderAnnotations(annotations) {
        const overlay = document.getElementById('annotations-overlay');
        if (!overlay) return;
        overlay.innerHTML = '';

        for (const ann of annotations) {
            const div = document.createElement('div');
            div.className = 'annotation-label';
            div.textContent = ann.text;

            // Project 3D position to screen
            if (ann.position) {
                const pos = new THREE.Vector3(ann.position[0], ann.position[1], ann.position[2]);
                pos.project(this.camera);
                const rect = this.canvas.getBoundingClientRect();
                const x = (pos.x * 0.5 + 0.5) * rect.width;
                const y = (-pos.y * 0.5 + 0.5) * rect.height;
                div.style.left = x + 'px';
                div.style.top = y + 'px';
                div.style.transform = 'translate(-50%, -50%)';
            }
            overlay.appendChild(div);
        }
    }

    _clearScene() {
        // Remove all non-light objects
        const toRemove = this.scene.children.filter(c => !c.isLight);
        toRemove.forEach(obj => {
            this.scene.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (Array.isArray(obj.material)) {
                    obj.material.forEach(m => m.dispose());
                } else {
                    obj.material.dispose();
                }
            }
        });
        this.objects.clear();
        this.labels = [];

        const overlay = document.getElementById('annotations-overlay');
        if (overlay) overlay.innerHTML = '';
    }

    _animate() {
        this.animationId = requestAnimationFrame(() => this._animate());

        const delta = this.clock.getDelta();
        const elapsed = this.clock.getElapsedTime();

        // Auto-rotate
        if (this.autoRotate && !this.isDragging) {
            this.cameraAngleX += delta * 0.3;
            this._updateCameraPosition();
        }

        // Animate objects
        for (const [id, entry] of this.objects) {
            const anim = entry.data.animation;
            if (!anim) continue;

            const mesh = entry.mesh;

            if (anim.type === 'orbit') {
                const speed = anim.speed || 1.0;
                const radius = anim.radius || 2.0;
                const angle = elapsed * speed;
                mesh.position.x = radius * Math.cos(angle);
                mesh.position.z = radius * Math.sin(angle);
            } else if (anim.type === 'rotate') {
                const speed = anim.speed || 1.0;
                if (anim.axis === 'y') mesh.rotation.y += delta * speed;
                else if (anim.axis === 'x') mesh.rotation.x += delta * speed;
                else if (anim.axis === 'z') mesh.rotation.z += delta * speed;
            } else if (anim.type === 'pulse') {
                const scale = 1 + 0.1 * Math.sin(elapsed * 3);
                mesh.scale.setScalar(scale);
            }
        }

        // Raycasting for hover
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const interactives = Array.from(this.objects.values())
            .filter(e => e.data.interactive)
            .map(e => e.mesh);
        const intersects = this.raycaster.intersectObjects(interactives);

        if (this.hoveredObject) {
            if (this.hoveredObject.material && this.hoveredObject.material.emissiveIntensity !== undefined) {
                this.hoveredObject.material.emissiveIntensity = this.hoveredObject.userData._origEmissive || 0;
            }
        }

        if (intersects.length > 0) {
            const obj = intersects[0].object;
            this.hoveredObject = obj;
            if (obj.material) {
                obj.userData._origEmissive = obj.material.emissiveIntensity;
                obj.material.emissive = obj.material.emissive || new THREE.Color(0x4444ff);
                obj.material.emissiveIntensity = 0.3;
            }
            this.canvas.style.cursor = 'pointer';
        } else {
            this.hoveredObject = null;
            this.canvas.style.cursor = 'grab';
        }

        // Update annotation positions
        const overlay = document.getElementById('annotations-overlay');
        if (overlay && overlay.children.length > 0) {
            // Annotations are re-projected each frame for dynamic scenes
        }

        this.renderer.render(this.scene, this.camera);
    }

    toggleAutoRotate() {
        this.autoRotate = !this.autoRotate;
        return this.autoRotate;
    }

    toggleLabels() {
        this.showLabels = !this.showLabels;
        for (const label of this.labels) {
            label.visible = this.showLabels;
        }
        return this.showLabels;
    }

    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        this._clearScene();
        this.renderer.dispose();
    }
}

// Export for use in app.js
window.SceneRenderer = SceneRenderer;
