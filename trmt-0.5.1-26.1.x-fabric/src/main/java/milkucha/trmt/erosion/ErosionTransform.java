package milkucha.trmt.erosion;

import milkucha.trmt.TRMTBlocks;
import milkucha.trmt.TRMTConfig;
import milkucha.trmt.block.ErodedDirtBlock;
import milkucha.trmt.block.ErodedGrassBlock;
import milkucha.trmt.block.ErodedSandBlock;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.DoublePlantBlock;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.properties.DoubleBlockHalf;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Shared erosion-transform logic used by both {@code MobEntityMixin} and
 * {@code ServerPlayerEntityMixin}. Extracted here so the transform chain and
 * vegetation-trampling rules only need to be maintained (and fixed) in one place,
 * instead of two near-identical copies drifting apart over time.
 */
public final class ErosionTransform {

    private ErosionTransform() {}

    /**
     * Maps a position rotation index (0–3, matching {@link BlockThresholds#posRotation})
     * to the corresponding {@link Direction} for the FACING block-state property.
     * 0 = SOUTH (0°), 1 = WEST (90° CW), 2 = NORTH (180°), 3 = EAST (270° CW).
     */
    public static Direction rotationToFacing(int rotation) {
        return switch (rotation) {
            case 1  -> Direction.WEST;
            case 2  -> Direction.NORTH;
            case 3  -> Direction.EAST;
            default -> Direction.SOUTH;
        };
    }

    /**
     * Checks whether trampled vegetation at {@code pos} has accumulated enough erosion
     * to break, and if so, destroys it (respecting double-plant halves and drop chance).
     */
    public static void tryBreakVegetation(Level world, ErosionMapManager manager,
                                           BlockPos pos, BlockState state) {
        ErosionEntry entry = manager.getChunkMap(ChunkPos.containing(pos)).getEntry(pos);
        if (entry == null || entry.getWalkedOnCount() < entry.getThreshold()) return;

        if (state.getBlock() instanceof DoublePlantBlock
                && state.getValue(DoublePlantBlock.HALF) == DoubleBlockHalf.LOWER) {
            BlockPos upper = pos.above();
            if (world.getBlockState(upper).is(state.getBlock())) {
                world.removeBlock(upper, false);
            }
            if (state.is(Blocks.TALL_GRASS)) {
                world.setBlock(pos, Blocks.SHORT_GRASS.defaultBlockState(), Block.UPDATE_ALL);
                manager.removeEntry(pos);
                return;
            }
        }

        float dropChance = TRMTConfig.get().erosionThresholds.vegetation.dropChance;
        boolean drops = dropChance >= 1.0f || (dropChance > 0.0f && ThreadLocalRandom.current().nextFloat() < dropChance);
        world.destroyBlock(pos, drops);
        manager.removeEntry(pos);
    }

    /**
     * Checks whether the block at {@code pos} has accumulated enough erosion to transform,
     * and if so, advances it to the next stage in the chain and clears its entry.
     */
    public static void tryTransform(Level world, ErosionMapManager manager, BlockPos pos) {
        BlockState state = world.getBlockState(pos);
        ErosionEntry entry = manager.getChunkMap(ChunkPos.containing(pos)).getEntry(pos);
        if (entry == null || entry.getWalkedOnCount() < entry.getThreshold()) return;

        if (state.is(Blocks.SAND)) {
            if (!world.getBlockState(pos.above()).isAir()) return;
            Direction erodedFacing = rotationToFacing(BlockThresholds.posRotation(pos));
            world.setBlock(pos,
                    TRMTBlocks.ERODED_SAND.defaultBlockState()
                            .setValue(ErodedSandBlock.FACING, erodedFacing)
                            .setValue(ErodedSandBlock.STAGE, 0),
                    Block.UPDATE_ALL);
            manager.removeEntry(pos);
            manager.writeCooldownEntry(pos, TRMTBlocks.ERODED_SAND, world.getGameTime());
            return;
        }

        if (state.is(TRMTBlocks.ERODED_SAND)) {
            if (!world.getBlockState(pos.above()).isAir()) return;
            int stage = state.getValue(ErodedSandBlock.STAGE);
            if (stage < 4) {
                world.setBlock(pos, state.setValue(ErodedSandBlock.STAGE, stage + 1), Block.UPDATE_ALL);
            }
            manager.removeEntry(pos);
            manager.writeCooldownEntry(pos, TRMTBlocks.ERODED_SAND, world.getGameTime());
            return;
        }

        if (BlockThresholds.isLeaves(state.getBlock())) {
            float dropChance = TRMTConfig.get().erosionThresholds.leaves.dropChance;
            boolean drops = dropChance >= 1.0f || (dropChance > 0.0f && ThreadLocalRandom.current().nextFloat() < dropChance);
            world.destroyBlock(pos, drops);
            manager.removeEntry(pos);
            return;
        }

        if (state.is(Blocks.GRASS_BLOCK)) {
            if (world.getBlockState(pos.above()).is(Blocks.WATER)) return;
            Direction erodedFacing = rotationToFacing(BlockThresholds.posRotation(pos));
            world.setBlock(pos,
                    TRMTBlocks.ERODED_GRASS_BLOCK.defaultBlockState()
                            .setValue(ErodedGrassBlock.FACING, erodedFacing)
                            .setValue(ErodedGrassBlock.STAGE, 0),
                    Block.UPDATE_ALL);
            manager.removeEntry(pos);
            manager.writeCooldownEntry(pos, TRMTBlocks.ERODED_GRASS_BLOCK, world.getGameTime());
            return;
        }

        if (state.is(TRMTBlocks.ERODED_GRASS_BLOCK)) {
            if (world.getBlockState(pos.above()).is(Blocks.WATER)) return;
            Direction facing = state.getValue(ErodedGrassBlock.FACING);
            int currentStage = state.getValue(ErodedGrassBlock.STAGE);
            if (currentStage < 4) {
                world.setBlock(pos, state.setValue(ErodedGrassBlock.STAGE, currentStage + 1), Block.UPDATE_ALL);
                manager.removeEntry(pos);
                manager.writeCooldownEntry(pos, TRMTBlocks.ERODED_GRASS_BLOCK, world.getGameTime());
                return;
            }
            world.setBlock(pos,
                    TRMTBlocks.ERODED_DIRT.defaultBlockState().setValue(ErodedDirtBlock.FACING, facing),
                    Block.UPDATE_ALL);
            manager.removeEntry(pos);
            return;
        }

        if (state.is(TRMTBlocks.ERODED_DIRT)) {
            if (world.getBlockState(pos.above()).is(Blocks.WATER)) return;
            Direction facing = state.getValue(ErodedDirtBlock.FACING);
            int currentStage = state.getValue(ErodedDirtBlock.STAGE);
            if (currentStage < 3) {
                world.setBlock(pos, state.setValue(ErodedDirtBlock.STAGE, currentStage + 1), Block.UPDATE_ALL);
                manager.removeEntry(pos);
                return;
            }
            world.setBlock(pos,
                    TRMTBlocks.ERODED_COARSE_DIRT.defaultBlockState().setValue(ErodedDirtBlock.FACING, facing),
                    Block.UPDATE_ALL);
            manager.removeEntry(pos);
            return;
        }

        if (!state.is(Blocks.DIRT)) return;
        if (world.getBlockState(pos.above()).is(Blocks.WATER)) return;
        Direction erodedFacing = rotationToFacing(BlockThresholds.posRotation(pos));
        world.setBlock(pos,
                TRMTBlocks.ERODED_DIRT.defaultBlockState()
                        .setValue(ErodedDirtBlock.FACING, erodedFacing)
                        .setValue(ErodedDirtBlock.STAGE, 1),
                Block.UPDATE_ALL);
        manager.removeEntry(pos);
    }
}
