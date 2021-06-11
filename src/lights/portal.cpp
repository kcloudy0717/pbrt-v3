
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

 // lights/infinite.cpp*
#include "lights/portal.h"
#include "imageio.h"
#include "paramset.h"
#include "sampling.h"
#include "stats.h"

namespace pbrt {

	// PortalInfiniteLight Method Definitions
	PortalInfiniteLight::PortalInfiniteLight(const Transform& LightToWorld,
		const Spectrum& L, int nSamples,
		const std::string& texmap,
		std::array<Point3f, 4> portal)
		: Light((int)LightFlags::Infinite, LightToWorld, MediumInterface(), nSamples)
		, portal(portal)
	{
		// Read texel data from _texmap_ and build mapping for accessing
		// the texels in _latlongmap_
		Point2i resolution;
		std::unique_ptr<RGBSpectrum[]> latlongTexels(nullptr);
		if (texmap != "") {
			latlongTexels = ReadImage(texmap, &resolution);
			if (latlongTexels)
				for (int i = 0; i < resolution.x * resolution.y; ++i)
					latlongTexels[i] *= L.ToRGBSpectrum();
		}
		if (!latlongTexels) {
			resolution.x = resolution.y = 1;
			latlongTexels = std::unique_ptr<RGBSpectrum[]>(new RGBSpectrum[1]);
			latlongTexels[0] = L.ToRGBSpectrum();
		}
		std::unique_ptr<MIPMap<RGBSpectrum>> latlongMap(
			new MIPMap<RGBSpectrum>(resolution, latlongTexels.get()));

		// Given a world space vector, this lambda returns the corresponding
		// RGBSpectrum from the spherical environment map.
		auto sphericalLookup = [&](const Vector3f& worldVec) {
			Vector3f lightVec = Normalize(WorldToLight(worldVec));

			// Convert to spherical coordinates for lookup in the
			// lat-long parameterized environment map
			Point2f stLatLong(SphericalPhi(lightVec) * Inv2Pi,
				SphericalTheta(lightVec) * InvPi);

			// MIPMap bilinearly interpolates the environment map at stLatLong
			RGBSpectrum L = Spectrum(latlongMap->Lookup(stLatLong), SpectrumType::Illuminant);

			return L;
		};

		// Initialize with a version of latlongMap resampled into rectified coordinates
		// as described in the paper.
		Vector3f p01 = Normalize(portal[1] - portal[0]);
		Vector3f p12 = Normalize(portal[2] - portal[1]);
		Vector3f p32 = Normalize(portal[2] - portal[3]);
		Vector3f p03 = Normalize(portal[3] - portal[0]);

		portalFrame = Frame::FromXY(p03, p01);

		std::vector<RGBSpectrum> rectifiedMap(latlongMap->Width() * latlongMap->Height());
		for (int y = 0; y < latlongMap->Height(); ++y)
		{
			for (int x = 0; x < latlongMap->Width(); ++x)
			{
				int index = y * latlongMap->Width() + x;

				// Resample _equalAreaImage_ to compute rectified image pixel $(x,y)$
				// Find $(u,v)$ coordinates in equal-area image for pixel
				Point2f uv((x + 0.5f) / latlongMap->Width(),
					(y + 0.5f) / latlongMap->Height());
				Vector3f w = ImageToWorld(uv);

				rectifiedMap[index] = sphericalLookup(w);
			}
		}

		Lmap.reset(new MIPMap<RGBSpectrum>({ latlongMap->Width(), latlongMap->Height() }, rectifiedMap.data()));

		// Initialize with a SATDistribution2D corresponding to scalar (luminance)
		// values from the rectified environment map to build the PDF for sampling.

		// Copied a portion from infinite.cpp
		int width = Lmap->Width(), height = Lmap->Height();
		std::unique_ptr<Float[]> img(new Float[width * height]);
		float fwidth = 0.5f / std::min(width, height);
		ParallelFor([&](int64_t v) {
			Float vp = (v + .5f) / (Float)height;
			for (int u = 0; u < width; ++u) {
				Float up = (u + .5f) / (Float)width;

				float duv_dw;
				ImageToWorld(Point2f(up, vp), &duv_dw);

				// Use luminance to create importance map, multipled by the jacobian determinant
				// for scaling
				img[u + v * width] = Lmap->Lookup(Point2f(up, vp), fwidth).y();
				img[u + v * width] *= duv_dw;
			}
		},
			height, 32);

		distribution.reset(new SATDistribution2D(img.get(), width, height));
	}

	// Copied from InfiniteAreaLight
	Spectrum PortalInfiniteLight::Power() const {
		return Pi * worldRadius * worldRadius *
			Spectrum(Lmap->Lookup(Point2f(.5f, .5f), .5f),
				SpectrumType::Illuminant);
	}

	Spectrum PortalInfiniteLight::Le(const RayDifferential& ray) const {
		Point2f uv = WorldToImage(Normalize(ray.d));
		return Spectrum(Lmap->Lookup(uv), SpectrumType::Illuminant);
	}

	Spectrum PortalInfiniteLight::Sample_Li(const Interaction& ref, const Point2f& u,
		Vector3f* wi, Float* pdf,
		VisibilityTester* vis) const {
		ProfilePhase _(Prof::LightSample);

		Bounds2f b = ImageBounds(ref.p);
		float mapPDF;
		Point2f uv = distribution->Sample(u, b, &mapPDF);
		if (mapPDF == 0) return Spectrum(0.f);

		// Convert portal image sample point to direction and compute PDF
		Float duv_dw;
		*wi = ImageToWorld(uv, &duv_dw);
		if (duv_dw == 0) return Spectrum(0.f);
		*pdf = mapPDF / duv_dw;
		CHECK(!std::isinf(*pdf));

		// Compute radiance for portal light sample and return _LightLiSample_
		*vis = VisibilityTester(ref, Interaction(ref.p + *wi * (2 * worldRadius),
			ref.time, mediumInterface));
		return Spectrum(Lmap->Lookup(uv), SpectrumType::Illuminant);
	}

	Float PortalInfiniteLight::Pdf_Li(const Interaction& ref, const Vector3f& w) const {
		ProfilePhase _(Prof::LightPdf);

		// Find image $(u,v)$ coordinates corresponding to direction _w_
		Float duv_dw;
		Point2f uv = WorldToImage(w, &duv_dw);
		if (duv_dw == 0)
			return 0;

		// Return PDF for sampling $(u,v)$ from reference point
		Bounds2f b = ImageBounds(ref.p);
		Float pdf = distribution->Pdf(uv, b);
		return pdf / duv_dw;
	}

	// Not necessary to implement for this assignment
	Spectrum PortalInfiniteLight::Sample_Le(const Point2f& u1, const Point2f& u2,
		Float time, Ray* ray, Normal3f* nLight,
		Float* pdfPos, Float* pdfDir) const {
		LOG(FATAL) << "Unimplemented";
		return Spectrum(0);
	}

	// Not necessary to implement for this assignment
	void PortalInfiniteLight::Pdf_Le(const Ray& ray, const Normal3f&, Float* pdfPos,
		Float* pdfDir) const {
		LOG(FATAL) << "Unimplemented";
	}

	static bool IsPortalPlanarQuad(const Point3f* P) {
		Vector3f p01 = Normalize(P[1] - P[0]);
		Vector3f p12 = Normalize(P[2] - P[1]);
		Vector3f p32 = Normalize(P[2] - P[3]);
		Vector3f p03 = Normalize(P[3] - P[0]);

		// Do opposite edges have the same direction?
		if (std::abs(Dot(p01, p32) - 1) > .001 ||
			std::abs(Dot(p12, p03) - 1) > .001)
			return false;
		// Sides perpendicular?
		if (std::abs(Dot(p01, p12)) > .001 ||
			std::abs(Dot(p12, p32)) > .001 ||
			std::abs(Dot(p32, p03)) > .001 ||
			std::abs(Dot(p03, p01)) > .001)
			return false;

		return true;
	}

	std::shared_ptr<PortalInfiniteLight> CreatePortalInfiniteLight(
		const Transform& light2world, const ParamSet& paramSet) {
		Spectrum L = paramSet.FindOneSpectrum("L", Spectrum(1.0));
		Spectrum sc = paramSet.FindOneSpectrum("scale", Spectrum(1.0));
		std::string texmap = paramSet.FindOneFilename("mapname", "");
		int nSamples = paramSet.FindOneInt("samples",
			paramSet.FindOneInt("nsamples", 1));
		if (PbrtOptions.quickRender) nSamples = std::max(1, nSamples / 4);

		int nVerts;
		const Point3f* P = paramSet.FindPoint3f("P", &nVerts);
		if (nVerts != 4)
			Error("Expected 4 values for portal \"P\" but got %d", nVerts);

		if (!IsPortalPlanarQuad(P))
			Error("Portal isn't a planar quadrilateral");

		return std::make_shared<PortalInfiniteLight>(light2world, L * sc, nSamples,
			texmap, std::array<Point3f, 4>{P[0], P[1], P[2], P[3]});
	}

}  // namespace pbrt
