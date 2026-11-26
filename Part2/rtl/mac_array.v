module mac_array #(
    parameter bw = 4,
    parameter psum_bw = 32,
    parameter col = 8,
    parameter row = 8
)(
    input clk, reset, compute_clear,
    input [row*bw-1:0] in_w,
    input [1:0] inst_w,
    input act_signed,
    output [psum_bw*col-1:0] out_s,
    output [col-1:0] valid
);
    wire [(row+1)*col*psum_bw-1:0] partial_chain;
    wire [row*col-1:0] valid_chain;
    reg [row*2-1:0] inst_w_temp;
    reg [row-1:0] signed_temp;
    assign partial_chain[col*psum_bw-1:0] = 0;
    assign out_s = partial_chain[row*col*psum_bw +: col*psum_bw];
    assign valid = valid_chain[(row-1)*col +: col];
    genvar r;
    generate for (r=0; r<row; r=r+1) begin : row_num
        mac_row #(.bw(bw), .psum_bw(psum_bw), .col(col)) mac_row_instance(
            .clk(clk), .reset(reset), .compute_clear(compute_clear),
            .in_w(in_w[r*bw +: bw]),
            .in_n(partial_chain[r*col*psum_bw +: col*psum_bw]),
            .out_s(partial_chain[(r+1)*col*psum_bw +: col*psum_bw]),
            .inst_w(inst_w_temp[r*2 +: 2]), .act_signed(signed_temp[r]),
            .valid(valid_chain[r*col +: col]));
    end endgenerate
    integer r_index;
    always @(posedge clk) begin
        if (reset || compute_clear) begin
            inst_w_temp <= 0;
            signed_temp <= 0;
        end else begin
            inst_w_temp[1:0] <= inst_w;
            signed_temp[0] <= act_signed;
            for (r_index=1; r_index<row; r_index=r_index+1) begin
                inst_w_temp[r_index*2 +: 2] <= inst_w_temp[(r_index-1)*2 +: 2];
                signed_temp[r_index] <= signed_temp[r_index-1];
            end
        end
    end
endmodule
