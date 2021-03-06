//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Pintér Tamás
// Neptun : JY4D5L
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	uniform vec3 wLookAt, wRight, wUp;          // pos of eye
	layout(location = 0) in vec2 cCamWindowVertex;	// Attrib Array 0
	out vec3 p;

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";
// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	const vec3		La = vec3(0.5f, 0.6f, 0.6f);
	const vec3		Le = vec3(0.8f, 0.8f, 0.8f);
	const vec3		lightPosition = vec3(0.4f, 0.4f, 0.25f);
	const vec3		ka = vec3(0.5f, 0.5f, 0.5f);
	const float		shininess = 500.0f;
	const int		maxdepth = 5;
	const float		epsilon = 0.01f;

	struct Hit {
		float t;
		vec3 position, normal;
		int mat;	//  smooth?
	};

	struct Ray {
		vec3 start, dir, weight;
	};

	const int		objFaces = 12;
	uniform int		top; // top paraboloid exists
	uniform vec3	wEye, v[20];
	uniform int		planes[objFaces * 3];
	uniform vec3	kd[2], ks[2], F0;

	void getObjPlane(int i, float scale, out vec3 p, out vec3 normal) {
		vec3 p1 = v[planes[3 * i] - 1], p2 = v[planes[3 * i + 1] - 1], p3 = v[planes[3 * i + 2] - 1];
		normal = cross(p2 - p1, p3 - p1);
		if (dot(p1, normal) < 0) normal = -normal; // normal vector should point outwards
		p = p1 * scale + vec3(0, 0, 0.03f);
	}

	Hit intersectConvexPolyhedron(Ray ray, Hit hit, float scale, int mat) {
		for(int i = 0; i < objFaces; i++) { // dodecahedron
			vec3 p1, normal;
			getObjPlane(i, scale, p1, normal);
			float ti = abs(dot(normal, ray.dir)) > epsilon ? dot(p1 - ray.start, normal) / dot(normal, ray.dir) : -1;
			if (ti <= epsilon || (ti > hit.t && hit.t > 0)) continue;
			vec3 pintersect = ray.start + ray.dir * ti;
			bool outside = false;
			for (int j = 0; j < objFaces; j++) { // check all other half spaces whether point is inside
				if (i == j) continue;
				vec3 p11, n;
				getObjPlane(j, scale, p11, n);
				if (dot(n, pintersect - p11) > 0) {
					outside = true;
					break;
				}
			}
			if (!outside) {
				hit.t = ti;
				hit.position = pintersect;
				hit.normal = normalize(normal);
				hit.mat = mat;
			}
		}
		return hit;
	}

	Hit solveQuadratic(float a, float b, float c, Ray ray, Hit hit, float zmin, float zmax, float normz) {
		float discr = b * b - 4.0f * a * c;
		if (discr >= 0) {
			float sqrt_discr = sqrt(discr);
			float t1 = (-b + sqrt_discr) / 2.0f / a;
			vec3 p = ray.start + ray.dir * t1;
			if (p.z > zmax || p.z < zmin) t1 = -1;
			float t2 = (-b - sqrt_discr) / 2.0f / a;
			p = ray.start + ray.dir * t2;
			if (p.z > zmax || p.z < zmin) t2 = -1;
			if (t2 > 0 && (t2 < t1 || t1 < 0)) t1 = t2;
			if (t1 > 0 && (t1 < hit.t || hit.t < 0)) {
				hit.t = t1;
				hit.position = ray.start + ray.dir * hit.t;
				hit.normal = normalize(vec3(-hit.position.x, -hit.position.y, normz));
				hit.mat = 2;
			}
		}
		return hit;
	}

	Hit intersectMirascope(Ray ray, Hit hit) {
		const float f = 0.25f; // focal distance
		const float H = 0.98f * f; // Height

		float a = dot(ray.dir.xy, ray.dir.xy);
		float b = dot(ray.dir.xy, ray.start.xy) * 2 - 4 * f * ray.dir.z;
		float c = dot(ray.start.xy, ray.start.xy) - 4 * f * ray.start.z;
		hit = solveQuadratic(a, b, c, ray, hit, 0, f/2, 2 * f);
		if (top == 0) return hit;
		b += 8 * f * ray.dir.z;
		c += 8 * f * ray.start.z - 4 * f * f;
		hit = solveQuadratic(a, b, c, ray, hit, f/2, H, -2 * f);
		return hit;
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1;
		bestHit = intersectMirascope(ray, bestHit);
		bestHit = intersectConvexPolyhedron(ray, bestHit, 0.02f, 0); // (ray, hit, scale, mat)
		bestHit = intersectConvexPolyhedron(ray, bestHit, 1.2f, 1);
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	vec3 trace(Ray ray) {
		vec3 outRadiance = vec3(0, 0, 0);
		for(int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) break;
			if (hit.mat < 2) {
				vec3 lightdir = normalize(lightPosition - hit.position);
				float cosTheta = dot(hit.normal, lightdir);
				if (cosTheta > 0) {
					vec3 LeIn = Le / dot(lightPosition - hit.position, lightPosition - hit.position);
					outRadiance += ray.weight * LeIn * kd[hit.mat] * cosTheta;
					vec3 halfway = normalize(-ray.dir + lightdir);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance += ray.weight * LeIn * ks[hit.mat] * pow(cosDelta, shininess);
				}
				ray.weight *= ka;
				break;
			}
			// mirror reflection
			ray.weight *= F0 + (vec3(1, 1, 1) - F0) * pow(dot(-ray.dir, hit.normal), 5);
			ray.start = hit.position + hit.normal * epsilon;
			ray.dir = reflect(ray.dir, hit.normal);
		}
		outRadiance += ray.weight + La;
		return outRadiance;
	}

	in vec3 p; // point on camera window corresponding to the pixel
	out vec4 fragmentColor;

	void main() {
		Ray ray;
		ray.start = wEye;
		ray.dir = normalize(p - wEye);
		fragmentColor = vec4(trace(ray), 1);
	}
)";

//---------------------------
struct Camera {
	//---------------------------
	vec3 eye, lookat, right, pvup, rvup;
	float fov = 45 * (float)M_PI / 180;

	Camera() : eye(0, 1, 1), pvup(0, 0, 1), lookat(0, 0, 0) { set(); }
	void set() {
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(pvup, w)) * f * tanf(fov / 2);
		rvup = normalize(cross(w, right)) * f * tanf(fov / 2);
	}
	void Animate(float t) {
		float r = sqrtf(eye.x * eye.x + eye.y * eye.y);
		eye = vec3(r * cos(t) + lookat.x, r * sin(t) + lookat.y, eye.z);
		set();
	}
	void Step(float step) {
		eye = normalize(eye + pvup * step) * length(eye);
		set();
	}
};

GPUProgram shader;
Camera camera;
bool animate = true;

float F(float n, float k) { return ((n - 1) * (n - 1) + k * k) / ((n + 1) * (n + 1) + k * k); }

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	unsigned int vao, vbo;
	glGenVertexArrays(1, &vao); glBindVertexArray(vao);
	glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
	float vertexCoords[] = { -1, -1, 1, -1, 1, 1, -1, 1 };
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

	shader.create(vertexSource, fragmentSource, "fragmentColor");
	shader.setUniform(1, "top");

	const float g = 0.618f, G = 1.618f;
	std::vector<vec3> v = {
		vec3(0, g, G), vec3(0, -g, G), vec3(0, -g, -G), vec3(0, g, -G),
		vec3(G, 0, g), vec3(-G, 0, g), vec3(-G, 0, -g), vec3(G, 0, -g),
		vec3(g, G, 0), vec3(-g, G, 0), vec3(-g, -G, 0), vec3(g, -G, 0),
		vec3(1, 1, 1), vec3(-1, 1, 1), vec3(-1, -1, 1), vec3(1, -1, 1),
		vec3(1, -1, -1), vec3(1, 1, -1), vec3(-1, 1, -1), vec3(-1, -1, -1),
	};

	for (int i = 0; i < v.size(); i++) shader.setUniform(v[i], "v[" + std::to_string(i) + "]");

	std::vector<int> planes = {
		1, 2, 16, 1, 13, 9, 1, 14, 6, 2, 15, 11, 3, 4, 18, 3, 17, 12, 3, 20, 7, 19, 10, 9, 16, 12, 17, 5, 8, 18, 14, 10, 19, 6, 7, 20,
	};

	for (int i = 0; i < planes.size(); i++) shader.setUniform(planes[i], "planes[" + std::to_string(i) + "]");

	shader.setUniform(vec3(0.1f, 0.2f, 0.3f), "kd[0]");
	shader.setUniform(vec3(1.5f, 0.6f, 0.4f), "kd[1]");
	shader.setUniform(vec3(5, 5, 5), "ks[0]");
	shader.setUniform(vec3(1, 1, 1), "ks[1]");
	shader.setUniform(vec3(F(0.17f, 3.1), F(0.35, 2.7), F(1.5, 1.9)), "F0");


}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	shader.setUniform(camera.eye, "wEye");
	shader.setUniform(camera.lookat, "wLookAt");
	shader.setUniform(camera.right, "wRight");
	shader.setUniform(camera.rvup, "wUp");
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glutSwapBuffers();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 't') shader.setUniform(1, "top");
	if (key == 'T') shader.setUniform(1, "top");
	if (key == 'f') camera.Step(0.1f);
	if (key == 'F') camera.Step(-0.1f);
	if (key == 'a') animate = !animate;
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	if (animate) camera.Animate(glutGet(GLUT_ELAPSED_TIME) / 1000);
	glutPostRedisplay();
}
