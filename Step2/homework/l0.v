// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission
module l0 (clk, in, out, rd, wr, o_full, reset, o_ready);
  parameter row = 8;
  parameter bw = 4;

  input clk, rd, wr, reset;
  input [row*bw-1:0] in;
  output [row*bw-1:0] out;
  output o_full, o_ready;

  wire [row-1:0] full, empty;
  reg [row-1:0] rd_en;

  assign o_full = |full;
  assign o_ready = ~|full;

  genvar i;
  for (i = 0; i < row; i = i + 1) begin : row_num
    fifo_depth64 #(.bw(bw)) fifo_instance (
      .rd_clk(clk), .wr_clk(clk), .reset(reset),
      .rd(rd_en[i]), .wr(wr),
      .in(in[i*bw +: bw]), .out(out[i*bw +: bw]),
      .o_full(full[i]), .o_empty(empty[i])
    );
  end

  always @(posedge clk) begin
    if (reset)
      rd_en <= {row{1'b0}};
    else
      // V1: all rows
      // rd_en <= {row{rd}};
      // V2: row 0 first
      rd_en <= (rd_en << 1) | rd;
  end
endmodule
