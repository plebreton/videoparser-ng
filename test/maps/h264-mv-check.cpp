// Compare the H.264 MV grid of the videoparser side data with FFmpeg's own
// motion vector export. Usage: h264-mv-check <file.mp4>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/motion_vector.h>
#include "libavcodec/videoparser_export.h"
}
#include <cstdio>

int main(int argc, char **argv) {
  AVFormatContext *fmt = nullptr;
  if (argc < 2 || avformat_open_input(&fmt, argv[1], nullptr, nullptr) < 0) return 2;
  avformat_find_stream_info(fmt, nullptr);
  int s = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  const AVCodec *codec = avcodec_find_decoder(fmt->streams[s]->codecpar->codec_id);
  AVCodecContext *ctx = avcodec_alloc_context3(codec);
  avcodec_parameters_to_context(ctx, fmt->streams[s]->codecpar);
  ctx->thread_count = 1;
  AVDictionary *opts = nullptr;
  av_dict_set(&opts, "flags2", "+export_mvs", 0);
  av_dict_set(&opts, "export_mv_matrix", "/dev/null", 0);
  if (avcodec_open2(ctx, codec, &opts) < 0) return 2;
  AVPacket *pkt = av_packet_alloc();
  AVFrame *frame = av_frame_alloc();
  long checked = 0, mismatches = 0, skipped = 0;
  auto check = [&]() {
    while (avcodec_receive_frame(ctx, frame) == 0) {
      const AVFrameSideData *mvs = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
      VPExportHeader *e = vp_export_get(frame);
      if (mvs && e && e->mv_w) {
        const AVMotionVector *v = (const AVMotionVector *)mvs->data;
        for (size_t i = 0; i < mvs->size / sizeof(*v); i++) {
          // dst is the center of the block; FFmpeg samples motion_val at the
          // top-left 4x4 cell of the block
          int cx = (v[i].dst_x - v[i].w / 2) / 4, cy = (v[i].dst_y - v[i].h / 2) / 4;
          if (cx < 0 || cy < 0 || cx >= e->mv_w || cy >= e->mv_h) continue;
          const VPExportMV *c = &vp_export_mv(e)[cy * e->mv_w + cx];
          int list = v[i].source > 0;
          int mx = list ? c->mv_l1_x : c->mv_l0_x, my = list ? c->mv_l1_y : c->mv_l0_y;
          // FFmpeg emits a vector for every sub-block of a macroblock for each
          // list that any of its sub-blocks uses (the macroblock-level mb_type);
          // a list unused by this sub-block reads as a zero vector. Skip only
          // that case: the list is used by another cell of the same macroblock.
          if (!(c->pred_flag & (1 << list))) {
            bool used_elsewhere = false;
            for (int yy = cy & ~3; yy < (cy & ~3) + 4 && !used_elsewhere; yy++)
              for (int xx = cx & ~3; xx < (cx & ~3) + 4; xx++)
                if (yy < e->mv_h && xx < e->mv_w &&
                    (vp_export_mv(e)[yy * e->mv_w + xx].pred_flag & (1 << list))) {
                  used_elsewhere = true;
                  break;
                }
            if (used_elsewhere && v[i].motion_x == 0 && v[i].motion_y == 0) { skipped++; continue; }
          }
          checked++;
          if (!(c->pred_flag & (1 << list)) || mx != v[i].motion_x || my != v[i].motion_y) {
            if (mismatches++ < 5)
              std::printf("frame %ld cell %d,%d list %d: export %d,%d (flag %d), ffmpeg %d,%d\n",
                          (long)ctx->frame_num, cx, cy, list, mx, my, c->pred_flag,
                          v[i].motion_x, v[i].motion_y);
          }
        }
      }
      av_frame_unref(frame);
    }
  };
  while (av_read_frame(fmt, pkt) >= 0) {
    if (pkt->stream_index == s) { avcodec_send_packet(ctx, pkt); check(); }
    av_packet_unref(pkt);
  }
  avcodec_send_packet(ctx, nullptr);
  check();
  std::printf("checked %ld motion vectors, %ld mismatches, %ld skipped (list unused by the sub-block)\n",
              checked, mismatches, skipped);
  return checked > 0 && mismatches == 0 ? 0 : 1;
}
