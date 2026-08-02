package milkucha.trmt.mixin;

import milkucha.trmt.TRMTBlocks;
import milkucha.trmt.TRMTConfig;
import milkucha.trmt.TRMTEffects;
import milkucha.trmt.erosion.BlockThresholds;
import milkucha.trmt.erosion.ErosionMapManager;
import milkucha.trmt.erosion.ErosionTransform;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.Mob;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Applies erosion to the block a leashed mob is standing on, mirroring the
 * player-erosion logic in {@link ServerPlayerEntityMixin}. The actual transform
 * chain and vegetation-trampling rules live in {@link ErosionTransform} so both
 * mixins share one implementation instead of maintaining duplicate copies.
 */
@Mixin(Mob.class)
public class MobEntityMixin {

    @Unique
    private BlockPos trmt$lastGroundPos = null;

    @Inject(method = "tick", at = @At("TAIL"))
    private void trmt$onTick(CallbackInfo ci) {
        Mob mob = (Mob)(Object) this;

        if (!(mob.level() instanceof ServerLevel)) return;

        if (!mob.onGround()) {
            trmt$lastGroundPos = null;
            return;
        }

        if (!mob.isLeashed()) return;

        if (mob.hasEffect(TRMTEffects.LIGHTNESS_ENTRY)) return;

        BlockPos groundPos = mob.blockPosition().below();

        Level world = mob.level();
        BlockState groundUpState = world.getBlockState(groundPos.above());
        if (groundUpState.is(TRMTBlocks.ERODED_SAND) || groundUpState.is(Blocks.SAND)) {
            groundPos = groundPos.above();
        }

        if (groundPos.equals(trmt$lastGroundPos)) return;
        trmt$lastGroundPos = groundPos.immutable();

        float mult = TRMTConfig.get().erosionMultipliers.player
                * TRMTConfig.get().erosionMultipliers.leash;

        ErosionMapManager manager = ErosionMapManager.getInstance();
        long gameTime = world.getGameTime();
        TRMTConfig.ErosionToggles erosion = TRMTConfig.get().erosion;

        BlockPos vegPos = groundPos.above();
        BlockState vegState = world.getBlockState(vegPos);
        if (erosion.vegetationEnabled && BlockThresholds.isVegetation(vegState.getBlock())) {
            manager.onStep(vegPos, vegState.getBlock(), 1.0f * mult, gameTime);
            ErosionTransform.tryBreakVegetation(world, manager, vegPos, vegState);
            manager.broadcastEntryUpdate(vegPos, vegState.getBlock());
        }

        BlockState state = world.getBlockState(groundPos);
        Block block = state.getBlock();

        boolean tracked = (erosion.grassEnabled && (state.is(Blocks.GRASS_BLOCK) || state.is(TRMTBlocks.ERODED_GRASS_BLOCK)))
                || (erosion.dirtEnabled && (state.is(Blocks.DIRT) || state.is(TRMTBlocks.ERODED_DIRT)))
                || (erosion.sandEnabled && (state.is(Blocks.SAND) || state.is(TRMTBlocks.ERODED_SAND)))
                || (erosion.leavesEnabled && BlockThresholds.isLeaves(block));

        if (!tracked) return;

        manager.onStep(groundPos, block, 1.0f * mult, gameTime);
        ErosionTransform.tryTransform(world, manager, groundPos);
        manager.broadcastEntryUpdate(groundPos, block);
    }
}
