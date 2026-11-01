/**
 * AQI Predictor — Frontend Application Logic
 * Handles form submission, API communication, result rendering, and animations.
 */

const API_BASE = 'http://localhost:5000/api';

// ====================================
// Particle Background
// ====================================
class ParticleSystem {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.particles = [];
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.init();
        this.animate();
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    init() {
        const count = Math.min(60, Math.floor(window.innerWidth / 25));
        this.particles = [];
        for (let i = 0; i < count; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 0.3,
                vy: (Math.random() - 0.5) * 0.3,
                size: Math.random() * 2 + 0.5,
                opacity: Math.random() * 0.3 + 0.1,
                hue: Math.random() > 0.5 ? 160 : 197
            });
        }
    }

    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.particles.forEach((p, i) => {
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0 || p.x > this.canvas.width) p.vx *= -1;
            if (p.y < 0 || p.y > this.canvas.height) p.vy *= -1;

            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            this.ctx.fillStyle = `hsla(${p.hue}, 80%, 60%, ${p.opacity})`;
            this.ctx.fill();

            // Draw connections
            for (let j = i + 1; j < this.particles.length; j++) {
                const p2 = this.particles[j];
                const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
                if (dist < 150) {
                    this.ctx.beginPath();
                    this.ctx.moveTo(p.x, p.y);
                    this.ctx.lineTo(p2.x, p2.y);
                    this.ctx.strokeStyle = `hsla(170, 60%, 50%, ${0.08 * (1 - dist / 150)})`;
                    this.ctx.lineWidth = 0.5;
                    this.ctx.stroke();
                }
            }
        });

        requestAnimationFrame(() => this.animate());
    }
}

// ====================================
// Navbar Scroll Effect
// ====================================
function initNavbar() {
    const navbar = document.getElementById('navbar');
    const navLinks = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // Smooth scroll for nav links
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.querySelector(link.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });
}

// ====================================
// Scroll Reveal Animation
// ====================================
function initScrollReveal() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.classList.add('visible');
                }, index * 100);
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.step-card, .scale-card').forEach(el => {
        observer.observe(el);
    });
}

// ====================================
// Sample Data Presets
// ====================================
const samplePresets = [
    { pm25: 12, pm10: 25, no2: 15, so2: 5, co: 0.5, o3: 30, temperature: 25, humidity: 55, wind_speed: 12, label: 'Good Air Quality' },
    { pm25: 35, pm10: 70, no2: 45, so2: 20, co: 2, o3: 60, temperature: 30, humidity: 65, wind_speed: 8, label: 'Moderate' },
    { pm25: 80, pm10: 150, no2: 80, so2: 50, co: 5, o3: 100, temperature: 35, humidity: 70, wind_speed: 4, label: 'Unhealthy' },
    { pm25: 180, pm10: 300, no2: 150, so2: 100, co: 10, o3: 160, temperature: 40, humidity: 80, wind_speed: 2, label: 'Very Unhealthy' },
    { pm25: 350, pm10: 500, no2: 250, so2: 200, co: 20, o3: 250, temperature: 42, humidity: 85, wind_speed: 1, label: 'Hazardous' },
];

let currentPresetIndex = 0;

function fillSampleData() {
    const preset = samplePresets[currentPresetIndex];
    
    document.getElementById('pm25').value = preset.pm25;
    document.getElementById('pm10').value = preset.pm10;
    document.getElementById('no2').value = preset.no2;
    document.getElementById('so2').value = preset.so2;
    document.getElementById('co').value = preset.co;
    document.getElementById('o3').value = preset.o3;
    document.getElementById('temperature').value = preset.temperature;
    document.getElementById('humidity').value = preset.humidity;
    document.getElementById('wind_speed').value = preset.wind_speed;

    // Animate inputs
    document.querySelectorAll('.input-card').forEach((card, i) => {
        card.style.transform = 'scale(0.97)';
        card.style.transition = 'transform 0.2s ease';
        setTimeout(() => {
            card.style.transform = 'scale(1)';
        }, 50 + i * 30);
    });

    currentPresetIndex = (currentPresetIndex + 1) % samplePresets.length;
}

// ====================================
// AQI Gauge Rendering
// ====================================
function updateGauge(aqiValue, color) {
    const maxAqi = 500;
    const fraction = Math.min(aqiValue / maxAqi, 1);
    
    // Gauge arc: from -90° (left) to +90° (right), center = 180° total
    const startAngle = Math.PI; // 180°
    const endAngle = startAngle + fraction * Math.PI; // up to 360°
    
    const cx = 100, cy = 100, r = 80;
    
    // Calculate end point of the arc
    const x = cx + r * Math.cos(endAngle);
    const y = cy + r * Math.sin(endAngle);
    
    const largeArcFlag = fraction > 0.5 ? 1 : 0;
    
    const arcPath = `M20 100 A80 80 0 ${largeArcFlag} 1 ${x.toFixed(2)} ${y.toFixed(2)}`;
    
    const gaugeArc = document.getElementById('gaugeArc');
    gaugeArc.setAttribute('d', arcPath);
    gaugeArc.setAttribute('stroke', color);
    
    // Update needle rotation: -90° (left) to +90° (right)
    const needleAngle = -90 + fraction * 180;
    const needle = document.getElementById('gaugeNeedle');
    needle.setAttribute('transform', `rotate(${needleAngle}, 100, 100)`);
}

// ====================================
// Render Results
// ====================================
function renderResults(data) {
    const { prediction, contributions, model_info } = data;
    
    // Show results panel
    document.getElementById('resultsEmpty').style.display = 'none';
    const resultsContent = document.getElementById('resultsContent');
    resultsContent.style.display = 'block';
    
    // Animate AQI number
    const aqiEl = document.getElementById('aqiValue');
    animateNumber(aqiEl, prediction.aqi, prediction.color);
    
    // Update gauge
    setTimeout(() => {
        updateGauge(prediction.aqi, prediction.color);
    }, 200);
    
    // Category badge
    const badge = document.getElementById('aqiBadge');
    badge.style.borderColor = prediction.color;
    badge.style.background = hexToRgba(prediction.color, 0.1);
    document.getElementById('aqiEmoji').textContent = prediction.emoji;
    document.getElementById('aqiCategory').textContent = prediction.category;
    document.getElementById('aqiCategory').style.color = prediction.color;
    
    // Health info
    const healthCard = document.getElementById('healthCard');
    healthCard.style.borderColor = hexToRgba(prediction.color, 0.3);
    document.getElementById('healthDescription').textContent = prediction.description;
    document.getElementById('healthAdvice').textContent = prediction.health_advice;
    document.getElementById('sensitiveAdvice').textContent = prediction.sensitive_advice;
    
    // Feature contributions
    const barsContainer = document.getElementById('contributionBars');
    barsContainer.innerHTML = '';
    
    const maxImportance = Math.max(...contributions.map(c => c.importance));
    
    contributions.forEach((contrib, index) => {
        const row = document.createElement('div');
        row.className = 'contrib-row';
        
        const percentage = (contrib.importance / maxImportance * 100).toFixed(0);
        
        row.innerHTML = `
            <span class="contrib-label">${contrib.feature}</span>
            <div class="contrib-bar-wrapper">
                <div class="contrib-bar" style="--target-width: ${percentage}%"></div>
            </div>
            <span class="contrib-value">${(contrib.importance * 100).toFixed(1)}%</span>
        `;
        
        barsContainer.appendChild(row);
        
        // Animate bar
        setTimeout(() => {
            row.querySelector('.contrib-bar').style.width = `${percentage}%`;
        }, 300 + index * 80);
    });
    
    // Model info
    document.getElementById('modelName').textContent = model_info.name;
    document.getElementById('modelR2').textContent = model_info.r2_score.toFixed(4);
    document.getElementById('modelMAE').textContent = model_info.mae.toFixed(2);
    
    // Scroll to results on mobile
    if (window.innerWidth <= 1024) {
        setTimeout(() => {
            document.getElementById('resultsPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 300);
    }
}

// ====================================
// Number Animation
// ====================================
function animateNumber(element, target, color) {
    const duration = 1000;
    const start = parseInt(element.textContent) || 0;
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing
        const ease = 1 - Math.pow(1 - progress, 4);
        const current = Math.round(start + (target - start) * ease);
        
        element.textContent = current;
        element.style.color = color;
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

// ====================================
// Utility Functions
// ====================================
function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// ====================================
// Toast Notifications
// ====================================
let toastTimeout;

function showToast(message) {
    const toast = document.getElementById('errorToast');
    document.getElementById('toastMessage').textContent = message;
    toast.classList.add('show');
    
    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => hideToast(), 5000);
}

function hideToast() {
    document.getElementById('errorToast').classList.remove('show');
}

// Make hideToast globally accessible for onclick
window.hideToast = hideToast;

// ====================================
// Form Submission
// ====================================
async function handleSubmit(e) {
    e.preventDefault();
    
    const btn = document.getElementById('predictBtn');
    btn.classList.add('loading');
    btn.disabled = true;
    
    const formData = {
        pm25: parseFloat(document.getElementById('pm25').value),
        pm10: parseFloat(document.getElementById('pm10').value),
        no2: parseFloat(document.getElementById('no2').value),
        so2: parseFloat(document.getElementById('so2').value),
        co: parseFloat(document.getElementById('co').value),
        o3: parseFloat(document.getElementById('o3').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        wind_speed: parseFloat(document.getElementById('wind_speed').value)
    };
    
    // Validate
    for (const [key, value] of Object.entries(formData)) {
        if (isNaN(value)) {
            showToast(`Please enter a valid number for ${key.replace('_', ' ')}`);
            btn.classList.remove('loading');
            btn.disabled = false;
            return;
        }
    }
    
    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Prediction failed');
        }
        
        if (data.success) {
            renderResults(data);
        } else {
            throw new Error(data.error || 'Unexpected response');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            showToast('Cannot connect to the server. Make sure the backend is running on port 5000.');
        } else {
            showToast(error.message);
        }
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}

// ====================================
// Reset Handler
// ====================================
function handleReset() {
    document.getElementById('resultsEmpty').style.display = 'flex';
    document.getElementById('resultsContent').style.display = 'none';
    
    // Reset gauge
    const gaugeArc = document.getElementById('gaugeArc');
    gaugeArc.setAttribute('d', 'M20 100 A80 80 0 0 1 20 100');
    gaugeArc.setAttribute('stroke', '#06d6a0');
    
    const needle = document.getElementById('gaugeNeedle');
    needle.setAttribute('transform', 'rotate(-90, 100, 100)');
    
    document.getElementById('aqiValue').textContent = '--';
    document.getElementById('aqiValue').style.color = '';
}

// ====================================
// Initialize Application
// ====================================
document.addEventListener('DOMContentLoaded', () => {
    // Init particle system
    const canvas = document.getElementById('particleCanvas');
    if (canvas) {
        new ParticleSystem(canvas);
    }
    
    // Init navbar
    initNavbar();
    
    // Init scroll reveal
    initScrollReveal();
    
    // Form handlers
    document.getElementById('aqiForm').addEventListener('submit', handleSubmit);
    document.getElementById('sampleBtn').addEventListener('click', fillSampleData);
    document.getElementById('resetBtn').addEventListener('click', handleReset);
    
    // Fetch model info on load
    fetchModelInfo();
});

// ====================================
// Fetch Model Info
// ====================================
async function fetchModelInfo() {
    try {
        const response = await fetch(`${API_BASE}/model-info`);
        if (response.ok) {
            const data = await response.json();
            if (data.success && data.model) {
                const r2 = (data.model.metrics.R2 * 100).toFixed(1);
                document.getElementById('modelAccuracy').textContent = `${r2}%`;
            }
        }
    } catch {
        // Server not running yet, that's okay
    }
}
