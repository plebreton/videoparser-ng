// Check that FrameInfo::maps equals the export files. Usage: maps-check <video>...
#include "VideoParser.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

struct Record { std::vector<int32_t> header; std::vector<char> data; };

static std::vector<Record> read_records(const char *path, int header_ints, size_t cell) {
  std::ifstream f(path, std::ios::binary);
  std::vector<Record> out;
  while (true) {
    Record r;
    r.header.resize(header_ints);
    if (!f.read(reinterpret_cast<char *>(r.header.data()), 4 * header_ints)) break;
    r.data.resize(size_t(r.header[1]) * r.header[2] * cell);
    f.read(r.data.data(), r.data.size());
    out.push_back(std::move(r));
  }
  return out;
}

int main(int argc, char **argv) {
  int failures = 0;
  for (int i = 1; i < argc; i++) {
    videoparser::OpenOptions o;
    o.block_maps = true;
    o.qp_export_path = "/tmp/maps-check-qp.bin";
    o.mv_export_path = "/tmp/maps-check-mv.bin";
    o.bits_export_path = "/tmp/maps-check-bits.bin";
    std::vector<videoparser::FrameMaps> maps;
    {
      videoparser::VideoParser parser(argv[i], o);
      videoparser::FrameInfo info;
      while (parser.parse_frame(info)) maps.push_back(info.maps);
    }
    auto qp = read_records("/tmp/maps-check-qp.bin", 3, 2);
    auto mv = read_records("/tmp/maps-check-mv.bin", 3, 12);
    auto bits = read_records("/tmp/maps-check-bits.bin", 4, 12);
    bool ok = maps.size() == qp.size() && maps.size() == mv.size() && maps.size() == bits.size();
    for (size_t k = 0; ok && k < maps.size(); k++) {
      const auto &m = maps[k];
      ok = m.qp_grid.width == qp[k].header[1] && m.qp_grid.height == qp[k].header[2] &&
           std::memcmp(m.qp.data(), qp[k].data.data(), qp[k].data.size()) == 0 &&
           m.mv_grid.width == mv[k].header[1] &&
           std::memcmp(m.mv.data(), mv[k].data.data(), mv[k].data.size()) == 0 &&
           m.bits_grid.width == bits[k].header[1] && m.bits_grid.cell_size == bits[k].header[3] &&
           std::memcmp(m.bits.data(), bits[k].data.data(), bits[k].data.size()) == 0;
      if (!ok) std::printf("%s: frame %zu differs\n", argv[i], k);
    }
    std::printf("%s: %zu frames, %s\n", argv[i], maps.size(), ok ? "ok" : "FAIL");
    if (!maps.empty())
      std::printf("%s: cell sizes qp %d mv %d bits %d\n", argv[i], maps[0].qp_grid.cell_size,
                  maps[0].mv_grid.cell_size, maps[0].bits_grid.cell_size);
    failures += !ok;
  }
  return failures ? 1 : 0;
}
