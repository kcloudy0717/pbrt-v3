
/*
	pbrt source code is Copyright(c) 1998-2016
						Matt Pharr, Greg Humphreys, and Wenzel Jakob.

	This file is part of pbrt.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are
	met:

	- Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.

	- Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
	IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
	TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
	PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
	HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
	SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
	LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
	DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
	THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_LIGHTS_PORTAL_H
#define PBRT_LIGHTS_PORTAL_H

 // lights/infinite.h*
#include "pbrt.h"
#include "light.h"
#include "texture.h"
#include "shape.h"
#include "scene.h"
#include "mipmap.h"
#include "satdistribution.h"

#include <array>
#include <vector>

namespace pbrt {
	template <typename T>
	inline constexpr T Sqr(T v) {
		return v * v;
	}

	// Frame Definition
	class Frame {
	public:
		// Frame Public Methods
		Frame() : x(1, 0, 0), y(0, 1, 0), z(0, 0, 1) {}
		Frame(Vector3f x, Vector3f y, Vector3f z) : x(x), y(y), z(z) {
			DCHECK_LT(std::abs(x.LengthSquared() - 1), 1e-4);
			DCHECK_LT(std::abs(y.LengthSquared() - 1), 1e-4);
			DCHECK_LT(std::abs(z.LengthSquared() - 1), 1e-4);
			DCHECK_LT(std::abs(Dot(x, y)), 1e-4);
			DCHECK_LT(std::abs(Dot(y, z)), 1e-4);
			DCHECK_LT(std::abs(Dot(z, x)), 1e-4);
		}

		static Frame FromXZ(Vector3f x, Vector3f z) { return Frame(x, Cross(z, x), z); }
		static Frame FromXY(Vector3f x, Vector3f y) { return Frame(x, y, Cross(x, y)); }

		static Frame FromZ(Vector3f z) {
			Vector3f x, y;
			CoordinateSystem(z, &x, &y);
			return Frame(x, y, z);
		}

		static Frame FromX(Vector3f x) {
			Vector3f y, z;
			CoordinateSystem(x, &y, &z);
			return Frame(x, y, z);
		}

		static Frame FromY(Vector3f y) {
			Vector3f x, z;
			CoordinateSystem(y, &z, &x);
			return Frame(x, y, z);
		}

		static Frame FromX(Normal3f x) {
			Vector3f y, z;
			CoordinateSystem(Vector3f(x), &y, &z);
			return Frame(Vector3f(x), y, z);
		}

		static Frame FromY(Normal3f y) {
			Vector3f x, z;
			CoordinateSystem(Vector3f(y), &z, &x);
			return Frame(x, Vector3f(y), z);
		}

		static Frame FromZ(Normal3f z) { return FromZ(Vector3f(z)); }

		Vector3f ToLocal(Vector3f v) const {
			return Vector3f(Dot(v, x), Dot(v, y), Dot(v, z));
		}

		Normal3f ToLocal(Normal3f n) const {
			return Normal3f(Dot(n, x), Dot(n, y), Dot(n, z));
		}

		Vector3f FromLocal(Vector3f v) const { return v.x * x + v.y * y + v.z * z; }

		Normal3f FromLocal(Normal3f n) const { return Normal3f(n.x * x + n.y * y + n.z * z); }

		// Frame Public Members
		Vector3f x, y, z;
	};

	// PortalInfiniteLight Declarations
	class PortalInfiniteLight : public Light {
	public:
		// PortalInfiniteLight Public Methods
		PortalInfiniteLight(const Transform& LightToWorld, const Spectrum& power,
			int nSamples, const std::string& texmap,
			std::array<Point3f, 4> portal);
		void Preprocess(const Scene& scene) {
			scene.WorldBound().BoundingSphere(&worldCenter, &worldRadius);
		}
		Spectrum Power() const;
		Spectrum Le(const RayDifferential& ray) const;
		Spectrum Sample_Li(const Interaction& ref, const Point2f& u, Vector3f* wi,
			Float* pdf, VisibilityTester* vis) const;
		Float Pdf_Li(const Interaction&, const Vector3f&) const;
		Spectrum Sample_Le(const Point2f& u1, const Point2f& u2, Float time,
			Ray* ray, Normal3f* nLight, Float* pdfPos,
			Float* pdfDir) const;
		void Pdf_Le(const Ray&, const Normal3f&, Float* pdfPos,
			Float* pdfDir) const;

	private:
		Point2f WorldToImage(Vector3f wWorld, Float* duv_dw = nullptr) const
		{
			Vector3f w = portalFrame.ToLocal(wWorld);

			// Compute Jacobian determinant of mapping $\roman{d}(u,v)/\roman{d}\omega$ if
			// needed
			if (duv_dw)
				*duv_dw = Sqr(Pi) * (1 - Sqr(w.x)) * (1 - Sqr(w.y)) / w.z;

			Float alpha = std::atan2(w.x, w.z), beta = std::atan2(w.y, w.z);
			DCHECK(!std::isnan(alpha + beta));
			return Point2f(Clamp((alpha + Pi / 2) / Pi, 0, 1),
				Clamp((beta + Pi / 2) / Pi, 0, 1));
		}

		Vector3f ImageToWorld(Point2f uv, Float* duv_dw = nullptr) const
		{
			Float alpha = -Pi / 2 + uv[0] * Pi, beta = -Pi / 2 + uv[1] * Pi;
			Float x = std::tan(alpha), y = std::tan(beta);
			DCHECK(!std::isinf(x) && !std::isinf(y));
			Vector3f w = Normalize(Vector3f(x, y, 1));
			// Compute Jacobian determinant of mapping $\roman{d}(u,v)/\roman{d}\omega$ if
			// needed
			if (duv_dw)
				*duv_dw = Sqr(Pi) * (1 - Sqr(w.x)) * (1 - Sqr(w.y)) / w.z;

			return portalFrame.FromLocal(w);
		}

		Bounds2f ImageBounds(const Point3f& p) const
		{
			Point2f p0 = WorldToImage(Normalize(portal[0] - p));
			Point2f p1 = WorldToImage(Normalize(portal[2] - p));
			return Bounds2f(p0, p1);
		}
	private:
		// PortalInfiniteLight Private Data
		std::unique_ptr<MIPMap<RGBSpectrum>> Lmap;
		Point3f worldCenter;
		Float worldRadius;
		Frame portalFrame;
		std::array<Point3f, 4> portal;
		std::unique_ptr<SATDistribution2D> distribution;
	};

	std::shared_ptr<PortalInfiniteLight> CreatePortalInfiniteLight(
		const Transform& light2world, const ParamSet& paramSet);

}  // namespace pbrt

#endif  // PBRT_LIGHTS_PORTAL_H
