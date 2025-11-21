# lab5/mc_tilecoder.py
import numpy as np


class TileCoder:
    """
    Tile coding for continuous state spaces.

    Designed to work with MountainCar-v0:
      state = [position, velocity]
    but is generic over number of dimensions.
    """

    def __init__(self, num_tilings, tiles_per_dim, state_bounds):
        """
        Args:
            num_tilings (int): Number of overlapping tilings (e.g., 8).
            tiles_per_dim (list[int]): Number of tiles per dimension
                for each tiling, e.g. [8, 8].
            state_bounds (list[tuple[float,float]]): (low, high) for
                each state dimension, usually taken from env.observation_space.
        """
        self.num_tilings = int(num_tilings)
        self.tiles_per_dim = np.asarray(tiles_per_dim, dtype=int)
        self.state_bounds = state_bounds
        self.num_dims = len(self.tiles_per_dim)

        # Width of a tile in each dimension
        self.tile_widths = np.array(
            [(high - low) / n_tiles
             for (low, high), n_tiles in zip(state_bounds, self.tiles_per_dim)],
            dtype=float,
        )

        # Asymmetric offsets for each tiling
        self.offsets = []
        for i in range(self.num_tilings):
            # Fraction of tile width to offset this tiling
            frac = i / self.num_tilings
            offset = frac * self.tile_widths
            self.offsets.append(offset)

        # Tiles per tiling and total features
        self.tiles_per_tiling = int(np.prod(self.tiles_per_dim))
        self.num_features = self.num_tilings * self.tiles_per_tiling

    # ---------- internal helpers ----------

    def _state_to_coords(self, state, offset):
        """
        Map continuous state to integer tile coordinates for a single tiling.
        """
        coords = []
        for dim in range(self.num_dims):
            s = state[dim]
            low, high = self.state_bounds[dim]

            # Shift by bounds and offset, then divide by tile width
            s_offset = s - low - offset[dim]
            tile = int(np.floor(s_offset / self.tile_widths[dim]))

            # Clip to valid range
            tile = np.clip(tile, 0, self.tiles_per_dim[dim] - 1)
            coords.append(tile)

        return coords

    # ---------- public API ----------

    def get_tiles(self, state):
        """
        Return indices of active tiles (one per tiling) for the given state.

        Returns:
            list[int]: length = num_tilings
        """
        state = np.asarray(state, dtype=float)
        tile_indices = []

        for tiling_idx, offset in enumerate(self.offsets):
            coords = self._state_to_coords(state, offset)

            # Flatten coords into a single index for this tiling.
            flat_index = 0
            multiplier = 1
            # Reverse order to mimic row-major flattening
            for k in range(self.num_dims - 1, -1, -1):
                flat_index += coords[k] * multiplier
                multiplier *= self.tiles_per_dim[k]

            global_index = tiling_idx * self.tiles_per_tiling + flat_index
            tile_indices.append(global_index)

        return tile_indices

    def get_features(self, state):
        """
        Return a sparse binary feature vector (NumPy array) with
        exactly num_tilings entries set to 1.0.
        """
        features = np.zeros(self.num_features, dtype=float)
        active_tiles = self.get_tiles(state)
        for idx in active_tiles:
            features[idx] = 1.0
        return features
