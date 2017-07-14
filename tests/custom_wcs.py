import galsim

# A pretty gratuitous custom WCS class to test the modules option in a piff config file.

class CustomWCS(galsim.config.WCSBuilder):

    def buildWCS(self, config, base, logger):
        index, index_key = galsim.config.GetIndex(config, base)
        if index == 0:
            return galsim.TanWCS(
                    galsim.AffineTransform(0.26, 0.05, -0.08, -0.24, galsim.PositionD(1024,1024)),
                    galsim.CelestialCoord(-5 * galsim.arcmin, -25 * galsim.degrees))
        elif index == 1:
            return galsim.TanWCS(
                    galsim.AffineTransform(0.25, -0.02, 0.01, 0.24, galsim.PositionD(1024,1024)),
                    galsim.CelestialCoord(5 * galsim.arcmin, -25 * galsim.degrees))
        else:
            raise ValueError("Custom WCS only supports building 2 WCS's")

galsim.config.RegisterWCSType('Custom', CustomWCS())
