/* eslint-disable import/no-extraneous-dependencies */
/* eslint-disable react-hooks/rules-of-hooks */
import { Stack } from '@chakra-ui/react';
import { useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { settingStyles } from './setting-styles';
import { useLive2dSettings } from '@/hooks/sidebar/setting/use-live2d-settings';
import { SwitchField } from './common';
import { Slider } from '@/components/ui/slider';

interface live2DProps {
  onSave?: (callback: () => void) => () => void
  onCancel?: (callback: () => void) => () => void
}

function live2D({ onSave, onCancel }: live2DProps): JSX.Element {
  const { t } = useTranslation();
  const {
    modelInfo,
    handleInputChange,
    handleSave,
    handleCancel,
  } = useLive2dSettings();

  useEffect(() => {
    if (!onSave || !onCancel) return;

    const cleanupSave = onSave(handleSave);
    const cleanupCancel = onCancel(handleCancel);

    return (): void => {
      cleanupSave?.();
      cleanupCancel?.();
    };
  }, [onSave, onCancel]);

  return (
    <Stack {...settingStyles.common.container}>
      <SwitchField
        label={t('settings.live2d.pointerInteractive')}
        checked={modelInfo.pointerInteractive ?? false}
        onChange={(checked) => handleInputChange('pointerInteractive', checked)}
      />

      <SwitchField
        label={t('settings.live2d.scrollToResize')}
        checked={modelInfo.scrollToResize ?? true}
        onChange={(checked) => handleInputChange('scrollToResize', checked)}
      />

      <Slider
        label={t('settings.live2d.petGazeGain')}
        showValue
        min={1}
        max={10}
        step={0.5}
        value={[modelInfo.petGazeGain ?? 5]}
        onValueChange={(e) => handleInputChange('petGazeGain', e.value[0])}
      />

      <Slider
        label={t('settings.live2d.petHeadOffsetY')}
        showValue
        min={-0.5}
        max={1.5}
        step={0.05}
        value={[modelInfo.petHeadOffsetY ?? 0.6]}
        onValueChange={(e) => handleInputChange('petHeadOffsetY', e.value[0])}
      />
    </Stack>
  );
}

export default live2D;
