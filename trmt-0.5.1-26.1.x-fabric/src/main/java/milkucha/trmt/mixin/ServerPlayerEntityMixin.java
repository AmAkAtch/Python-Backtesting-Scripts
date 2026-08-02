package milkucha.trmt.mixin;

import milkucha.trmt.TRMTBlocks;
import milkucha.trmt.TRMTConfig;
import milkucha.trmt.TRMTEffects;
import milkucha.trmt.erosion.BlockThresholds;
import milkucha.trmt.erosion.ErosionMapManager;
import milkucha.trmt.erosion.ErosionTransform;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(ServerPlayer.class)
public class ServerPlayerEntityMixin {


    /** Last block position this player was standing on. Null while airborne. */
    @Unique
    private BlockPos trmt$lastGroundPos = null;

    @Inject(method = "tick", at = @At("TAIL"))
    private void trmt$onTick(CallbackInfo ci) {
        ServerPlayer player = (ServerPlayer) (Object) this;

        // Determine whether the player is mounted and, if so, delegate ground detection to the vehicle.
        Entity vehicle = player.getVehicle();
        boolean mounted = vehicle != null;
        boolean onGround = mounted ? vehicle.onGround() : player.onGround();

        if (!onGround) {
            // Airborne (or vehicle airborne) — clear last ground position so the next landing registers.
            trmt$lastGroundPos = null;
            return;
        }

        // getBlockPos() returns the block at the entity's Y coordinate (feet level).
        // The block they are *standing on* is one below.
        BlockPos groundPos = (mounted ? vehicle.blockPosition() : player.blockPosition()).below();

        // Sunken blocks (e.g. ERODED_SAND stages 1–4) have a collision height < 1, so the
        // player's feet land inside the block space and getBlockPos().down() resolves one block
        // too low. Correct by checking one block up when groundPos yields nothing tracked.
        Level world = player.level();
        BlockState groundUpState = world.getBlockState(groundPos.above());
        if (groundUpState.is(TRMTBlocks.ERODED_SAND) || groundUpState.is(Blocks.SAND)) {
            groundPos = groundPos.above();
        }

        // Only process when the player (or vehicle) moves onto a new block, not while standing still.
        if (groundPos.equals(trmt$lastGroundPos)) {
            return;
        }

        trmt$lastGroundPos = groundPos.immutable();

        if (player.isShiftKeyDown()) return;

        // Potion of Lightness suppresses erosion for the affected player or their mount.
        if (!mounted && player.hasEffect(TRMTEffects.LIGHTNESS_ENTRY)) return;
        if (vehicle instanceof LivingEntity livingVehicle
                && livingVehicle.hasEffect(TRMTEffects.LIGHTNESS_ENTRY)) return;

        BlockState state = world.getBlockState(groundPos);
        Block block = state.getBlock();

        // Transformation chain:
        //   grass_block ──► eroded_grass_block (s0→s4) ──► eroded_dirt (s0→s3) ──► eroded_coarse_dirt (final)
        //   dirt ────────► eroded_dirt (s1→s3) ──► eroded_coarse_dirt (final)
        // Apply player erosion multiplier; mounted players get an additional configurable boost.
        float mult = TRMTConfig.get().erosionMultipliers.player
                * (mounted ? TRMTConfig.get().erosionMultipliers.mounted : 1.0f);

        ErosionMapManager manager = ErosionMapManager.getInstance();
        long gameTime = world.getGameTime();
        TRMTConfig.ErosionToggles erosion = TRMTConfig.get().erosion;

        // Check for vegetation at the player's feet level (one block above the ground).
        // Vegetation has no collision so the player passes through it — track and break it.
        // This fires regardless of what the ground block is so that vegetation on any surface
        // can be trampled, even when the ground block's own erosion category is disabled.
        BlockPos vegPos = groundPos.above();
        BlockState vegState = world.getBlockState(vegPos);
        if (erosion.vegetationEnabled && BlockThresholds.isVegetation(vegState.getBlock())) {
            manager.onStep(vegPos, vegState.getBlock(), 1.0f * mult, gameTime);
            ErosionTransform.tryBreakVegetation(world, manager, vegPos, vegState);
            manager.broadcastEntryUpdate(vegPos, vegState.getBlock());
        }

        boolean tracked = (erosion.grassEnabled && (state.is(Blocks.GRASS_BLOCK) || state.is(TRMTBlocks.ERODED_GRASS_BLOCK)))
                || (erosion.dirtEnabled && (state.is(Blocks.DIRT) || state.is(TRMTBlocks.ERODED_DIRT)))
                || (erosion.sandEnabled && (state.is(Blocks.SAND) || state.is(TRMTBlocks.ERODED_SAND)))
                || (erosion.leavesEnabled && BlockThresholds.isLeaves(block));

        if (!tracked) {
            return;
        }

        manager.onStep(groundPos, block, 1.0f * mult, gameTime);
        ErosionTransform.tryTransform(world, manager, groundPos);
        manager.broadcastEntryUpdate(groundPos, block);

        // Spread erosion to adjacent blocks based on the player's facing direction.
        // Front (the direction the player faces): +0.2
        // Left and right: +0.5 each
        // Back: nothing
        Direction facing = player.getDirection();
        Direction left  = facing.getCounterClockWise();
        Direction right = facing.getClockWise();

        trmt$stepAdjacent(world, manager, groundPos.relative(facing), 0.2f * mult, gameTime);
        trmt$stepAdjacent(world, manager, groundPos.relative(left),   0.5f * mult, gameTime);
        trmt$stepAdjacent(world, manager, groundPos.relative(right),  0.5f * mult, gameTime);
    }

    @Unique
    private static void trmt$stepAdjacent(Level world, ErosionMapManager manager,
                                           BlockPos pos, float amount, long gameTime) {
        BlockState adjState = world.getBlockState(pos);
        TRMTConfig.ErosionToggles erosion = TRMTConfig.get().erosion;
        if ((erosion.grassEnabled && (adjState.is(Blocks.GRASS_BLOCK) || adjState.is(TRMTBlocks.ERODED_GRASS_BLOCK)))
                || (erosion.dirtEnabled && (adjState.is(Blocks.DIRT) || adjState.is(TRMTBlocks.ERODED_DIRT)))
                || (erosion.sandEnabled && (adjState.is(Blocks.SAND) || adjState.is(TRMTBlocks.ERODED_SAND)))
                || (erosion.leavesEnabled && BlockThresholds.isLeaves(adjState.getBlock()))) {
            manager.onStep(pos, adjState.getBlock(), amount, gameTime);
            ErosionTransform.tryTransform(world, manager, pos);
        }
    }
}
