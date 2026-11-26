// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission
module mac_row (clk, out_s, in_w, in_n, valid, inst_w, reset);
  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;

  input clk, reset;
  input [bw-1:0] in_w;
  input [psum_bw*col-1:0] in_n;
  input [1:0] inst_w;
  output [psum_bw*col-1:0] out_s;
  output [col-1:0] valid;

  wire [(col+1)*bw-1:0] temp;
  wire [2*(col+1)-1:0] inst_temp;
  assign temp[bw-1:0] = in_w;
  assign inst_temp[1:0] = inst_w;

  genvar i;
  for (i = 1; i <= col; i = i + 1) begin : col_num
    localparam WEST = i - 1;
    mac_tile #(.bw(bw), .psum_bw(psum_bw)) mac_tile_instance (
      .clk(clk),
      .reset(reset),
      .in_w(temp[WEST*bw +: bw]),
      .out_e(temp[i*bw +: bw]),
      .inst_w(inst_temp[2*WEST +: 2]),
      .inst_e(inst_temp[2*i +: 2]),
      .in_n(in_n[WEST*psum_bw +: psum_bw]),
      .out_s(out_s[WEST*psum_bw +: psum_bw])
    );
    assign valid[WEST] = inst_temp[2*i + 1];
  end
endmodule
