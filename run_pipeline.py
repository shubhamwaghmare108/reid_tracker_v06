#!/usr/bin/env python
"""Command-line entry point for processing a webcam, video file, or stream."""
import sys
from pathlib import Path
from app.config import Settings
from app.core.pipeline import ReIDPipeline
from app.utils.pipeline_logging import configure_pipeline_logging, get_pipeline_logger

if __name__ == '__main__':
    try:
        settings = Settings()
        log_path = configure_pipeline_logging(settings.output_dir)
        logger = get_pipeline_logger()

        # Camera indexes are integers (0 is normally the default camera); other values are paths/URLs.
        source_arg = sys.argv[1] if len(sys.argv) > 1 else '0'
        source = int(source_arg) if source_arg.isdecimal() else source_arg
        output = settings.output_dir / 'output_video.mp4'
        logger.info('Starting pipeline command: source=%s, log=%s', source, log_path)
        pipeline = ReIDPipeline(settings)
        pipeline.run_on_video(source, save_to_db=True, output_video=output)
        print(f'Pipeline completed successfully. Log: {log_path}')
    except KeyboardInterrupt:
        print('Pipeline stopped by user.', file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        logger = get_pipeline_logger()
        logger.exception('Pipeline command failed.')
        print(f'Pipeline failed: {error}', file=sys.stderr)
        print('See output/pipeline.log for details.', file=sys.stderr)
        sys.exit(1)
